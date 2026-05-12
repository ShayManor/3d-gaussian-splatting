from logging import log, WARNING, INFO, DEBUG

import cv2
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

from src.video.calibrate import Calibrator


class VideoSFM:
    """
    Class representing one video and getting the structure from frames
    """

    def __init__(self, device="cuda", matcher="sift", focal_px=None, focal_35mm=None):
        self.calibrator = Calibrator(matcher, focal_px=focal_px, focal_35mm=focal_35mm)


    def process_video_frames(self, frames, video_path, stride=5):
        """
        Run incremental SfM over a series of frames.

        First accepted pair is initialized from the essential matrix (sets metric
        scale, with |t|=1 by recoverPose convention). Every accepted frame after
        that is registered with PnP against world points already triangulated by
        prior pairs — so the global scale stays consistent and there is no scale
        drift across pairs. New world points are triangulated using the PnP-recovered
        absolute pose for the current frame and the previous accepted pose.
        """
        # Environment probe: surface anything obviously wrong (no SIFT support,
        # blank frames, dtype mismatch) before we run 521 pairs and report 0
        # matches with no actionable signal. Runs once.
        self._log_environment_probe(frames)

        K = self.calibrator.identify_intrinsics(
            frames[: min(50, len(frames))], video_path
        )

        frame_indices = list(range(0, len(frames), stride))

        if not frame_indices:
            return {
                "poses": np.empty((0, 4, 4)),
                "intrinsics": K,
                "points_3d": np.empty((0, 3)),
                "colors": np.empty((0, 3)),
                "frame_indices": np.empty((0,), int),
            }

        poses = [np.identity(4)]
        pose_frame_indices = [frame_indices[0]]
        points_3d = []
        point_colors = []

        # Track table: 2D positions (in the LAST accepted frame) of points whose
        # world index we know. Used to find 3D-2D correspondences for PnP.
        prev_track_kp = np.empty((0, 2), dtype=np.float32)
        prev_track_widx = np.empty((0,), dtype=np.int64)

        prev_frame_idx = frame_indices[0]
        first_pair_done = False

        # Per-gate skip counters so the user can see which gate is the bottleneck.
        skips = {
            "too_similar": 0,
            "few_matches": 0,
            "pose_recovery_failed": 0,
            "low_inliers_or_flow": 0,
            "triangulation_empty": 0,
            "triangulation_too_few": 0,
            "klt_failed": 0,
            "pnp_too_few_tracks": 0,
            "pnp_failed": 0,
            "trivial_motion": 0,
        }

        # Gates tuned for SIFT (sparse, conservative). LoFTR/dense matchers
        # comfortably exceed these on the first pair too.
        MIN_MATCHES = 8
        MIN_INLIERS = 30
        MIN_FLOW_PX = 4.0
        MIN_PNP_TRACKS = 8
        # KLT optical-flow tracking carries sub-pixel positions; 6 px guards
        # against drift in the LoFTR-grid case where match positions also wobble.
        TRACK_RADIUS_PX = 6.0
        # Tighter reprojection (was 2.0) to reject near-infinity triangulations
        # that survive cheirality but inflate scene_extent downstream.
        MAX_REPROJ_PX = 1.0

        # Diagnostic: collect raw match counts so the user can see the actual
        # distribution when most pairs fail.
        match_counts = []

        for i, frame_idx in enumerate(
            tqdm(frame_indices[1:], total=len(frame_indices) - 1, desc="Processing pairs"), 1
        ):
            if self._too_similar(frames[prev_frame_idx], frames[frame_idx]):
                skips["too_similar"] += 1
                continue

            matches = self.calibrator.extract_all_matches(
                [frames[prev_frame_idx], frames[frame_idx]]
            )
            n_match = len(matches[0]["pts1"]) if matches else 0
            match_counts.append(n_match)
            if n_match < MIN_MATCHES:
                skips["few_matches"] += 1
                continue

            pts1, pts2 = matches[0]["pts1"], matches[0]["pts2"]

            if not first_pair_done:
                R_rel, t_rel, inliers = self.estimate_pose_from_matches(pts1, pts2, K)
                if R_rel is None:
                    skips["pose_recovery_failed"] += 1
                    continue

                ninl = int(np.count_nonzero(inliers))
                flow_med = (
                    float(np.median(np.linalg.norm(pts2[inliers] - pts1[inliers], axis=1)))
                    if ninl else 0.0
                )
                if ninl < MIN_INLIERS or flow_med < MIN_FLOW_PX:
                    skips["low_inliers_or_flow"] += 1
                    continue

                pts1_in = pts1[inliers]
                pts2_in = pts2[inliers]

                rel_pose = np.eye(4)
                rel_pose[:3, :3] = R_rel
                rel_pose[:3, 3] = t_rel.squeeze()
                # poses[0] is identity, so absolute world-to-cam is rel_pose.
                absolute_pose = rel_pose @ poses[-1]

                tri = self._triangulate_pair(
                    pts1_in, pts2_in, K, poses[-1], absolute_pose, MAX_REPROJ_PX
                )
                if tri is None:
                    skips["triangulation_empty"] += 1
                    continue
                X_world, kept = tri
                if len(X_world) < MIN_INLIERS // 2:
                    skips["triangulation_too_few"] += 1
                    continue

                start = len(points_3d)
                points_3d.extend(X_world)
                point_colors.extend(self._extract_point_colors(frames[frame_idx], pts2_in[kept]))
                poses.append(absolute_pose)
                pose_frame_indices.append(frame_idx)

                prev_track_kp = pts2_in[kept].astype(np.float32).copy()
                prev_track_widx = np.arange(start, start + len(X_world), dtype=np.int64)
                prev_frame_idx = frame_idx
                first_pair_done = True
                continue

            # PnP path. Carry existing tracks forward with KLT (pixel-accurate,
            # detector-independent), then run PnP on the surviving 3D-2D pairs.
            gray_prev = cv2.cvtColor(frames[prev_frame_idx], cv2.COLOR_BGR2GRAY)
            gray_curr = cv2.cvtColor(frames[frame_idx], cv2.COLOR_BGR2GRAY)
            kp_in = prev_track_kp.reshape(-1, 1, 2).astype(np.float32)
            kp_out, st, _ = cv2.calcOpticalFlowPyrLK(
                gray_prev, gray_curr, kp_in, None,
                winSize=(15, 15), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-3),
            )
            if kp_out is None or st is None:
                skips["klt_failed"] += 1
                continue
            st = st.reshape(-1).astype(bool)
            kp_out = kp_out.reshape(-1, 2)

            # Reject tracks that left the image
            h, w = gray_curr.shape
            in_img = (kp_out[:, 0] >= 0) & (kp_out[:, 0] < w) & (kp_out[:, 1] >= 0) & (kp_out[:, 1] < h)
            valid = st & in_img

            if int(valid.sum()) < MIN_PNP_TRACKS:
                skips["pnp_too_few_tracks"] += 1
                continue

            tracked_widx = prev_track_widx[valid]
            tracked_2d_curr = kp_out[valid]
            obj_pts = np.asarray(points_3d, dtype=np.float64)[tracked_widx].reshape(-1, 1, 3)
            img_pts = tracked_2d_curr.astype(np.float64).reshape(-1, 1, 2)

            ok, rvec, tvec, pnp_inl = cv2.solvePnPRansac(
                objectPoints=obj_pts,
                imagePoints=img_pts,
                cameraMatrix=K,
                distCoeffs=None,
                iterationsCount=200,
                reprojectionError=MAX_REPROJ_PX,
                confidence=0.999,
                flags=cv2.SOLVEPNP_EPNP,
            )
            if not ok or pnp_inl is None or len(pnp_inl) < MIN_PNP_TRACKS:
                skips["pnp_failed"] += 1
                continue

            pnp_inl = pnp_inl.ravel()
            rvec, tvec = cv2.solvePnPRefineLM(
                obj_pts[pnp_inl], img_pts[pnp_inl], K, None, rvec, tvec
            )
            R_w2c, _ = cv2.Rodrigues(rvec)
            absolute_pose = np.eye(4)
            absolute_pose[:3, :3] = R_w2c
            absolute_pose[:3, 3] = tvec.ravel()

            # Sanity: motion since the previous accepted pose must be non-trivial.
            T_rel = absolute_pose @ np.linalg.inv(poses[-1])
            if np.linalg.norm(T_rel[:3, 3]) < 1e-3:
                skips["trivial_motion"] += 1
                continue

            # Discover new points via LoFTR matches in this pair, excluding any
            # match whose pts1 lands close to an already-tracked keypoint in the
            # previous frame (so we don't re-triangulate what we already track).
            tree = cKDTree(prev_track_kp)
            d_lookup, _ = tree.query(pts1, distance_upper_bound=TRACK_RADIUS_PX)
            new_match_mask = ~(np.isfinite(d_lookup) & (d_lookup < TRACK_RADIUS_PX))

            new_kp_curr = np.empty((0, 2), dtype=np.float32)
            new_widx = np.empty((0,), dtype=np.int64)
            if int(new_match_mask.sum()) >= MIN_PNP_TRACKS:
                pts1_new = pts1[new_match_mask]
                pts2_new = pts2[new_match_mask]
                tri = self._triangulate_pair(
                    pts1_new, pts2_new, K, poses[-1], absolute_pose, MAX_REPROJ_PX
                )
                if tri is not None:
                    X_world_new, kept_new = tri
                    start = len(points_3d)
                    points_3d.extend(X_world_new)
                    point_colors.extend(
                        self._extract_point_colors(frames[frame_idx], pts2_new[kept_new])
                    )
                    new_kp_curr = pts2_new[kept_new].astype(np.float32)
                    new_widx = np.arange(start, start + len(X_world_new), dtype=np.int64)

            poses.append(absolute_pose)
            pose_frame_indices.append(frame_idx)

            # Next track table = surviving KLT tracks (PnP inliers, with their
            # KLT-tracked 2D position in the just-registered frame) + new points.
            carried_2d_curr = tracked_2d_curr[pnp_inl].astype(np.float32)
            carried_widx = tracked_widx[pnp_inl]
            prev_track_kp = np.vstack([carried_2d_curr, new_kp_curr]) if (len(carried_2d_curr) + len(new_kp_curr)) else np.empty((0, 2), dtype=np.float32)
            prev_track_widx = np.concatenate([carried_widx, new_widx])
            prev_frame_idx = frame_idx

        n_candidates = max(len(frame_indices) - 1, 1)
        n_skipped = sum(skips.values())
        log(
            INFO,
            f"SFM complete: {len(poses)} poses, {len(points_3d)} 3D points "
            f"(skipped {n_skipped}/{n_candidates} candidates)",
        )
        if n_skipped > 0:
            breakdown = ", ".join(f"{k}={v}" for k, v in skips.items() if v > 0)
            level = WARNING if (len(poses) < 2 or n_skipped > 0.5 * n_candidates) else INFO
            log(level, f"Skip breakdown: {breakdown}")
        if match_counts:
            mc = np.array(match_counts)
            log(
                INFO,
                f"Match count distribution over {len(mc)} attempted pairs: "
                f"min={mc.min()} p25={int(np.percentile(mc,25))} "
                f"median={int(np.median(mc))} p75={int(np.percentile(mc,75))} "
                f"max={mc.max()} (gate MIN_MATCHES={MIN_MATCHES})",
            )

        poses_arr = np.array(poses)
        return {
            "poses": poses_arr,
            "intrinsics": K,
            "points_3d": np.array(points_3d) if points_3d else np.empty((0, 3)),
            "colors": np.array(point_colors) if point_colors else np.empty((0, 3)),
            "frame_indices": np.array(pose_frame_indices, dtype=int),
        }

    def _log_environment_probe(self, frames):
        """
        Print diagnostics that quickly distinguish "thresholds need tuning" from
        "OpenCV/matcher is broken" or "frames decoded wrong". Runs once per call.
        """
        if not frames:
            log(WARNING, "[probe] no frames passed to process_video_frames")
            return

        f0 = frames[0]
        log(
            INFO,
            f"[probe] OpenCV {cv2.__version__}  matcher_type={self.calibrator.matcher_type}  "
            f"frame[0] shape={f0.shape} dtype={f0.dtype} mean={float(f0.mean()):.1f} std={float(f0.std()):.1f}",
        )

        # Pick a second frame ~5% in for the pair-test.
        j = min(len(frames) - 1, max(1, len(frames) // 20)) if len(frames) > 1 else None

        if self.calibrator.matcher_type in ("sift", "opencv") and self.calibrator.alg is not None:
            try:
                gray0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
                kp0, des0 = self.calibrator.alg.detectAndCompute(gray0, None)
                n_kp0 = 0 if kp0 is None else len(kp0)
                des_shape = None if des0 is None else des0.shape
                log(
                    INFO,
                    f"[probe] {self.calibrator.matcher_type} on frame[0]: "
                    f"keypoints={n_kp0}, des shape={des_shape}",
                )
                if j is not None:
                    gray1 = cv2.cvtColor(frames[j], cv2.COLOR_BGR2GRAY)
                    kp1, des1 = self.calibrator.alg.detectAndCompute(gray1, None)
                    n_kp1 = 0 if kp1 is None else len(kp1)
                    if des0 is not None and des1 is not None and n_kp0 > 0 and n_kp1 > 0:
                        raw = self.calibrator.matcher.knnMatch(des0, des1, k=2)
                        n_ratio = sum(
                            1 for p in raw if len(p) == 2 and p[0].distance < 0.75 * p[1].distance
                        )
                        log(
                            INFO,
                            f"[probe] frame[0] vs frame[{j}]: kp1={n_kp1}, "
                            f"raw knn={len(raw)}, ratio<0.75 = {n_ratio}",
                        )
                    else:
                        log(WARNING, f"[probe] frame[0] vs frame[{j}]: cannot match (kp1={n_kp1})")
            except Exception as e:
                log(WARNING, f"[probe] detector test raised {type(e).__name__}: {e}")

        elif self.calibrator.matcher_type == "loftr" and j is not None:
            try:
                pts0, pts1 = self.calibrator.match_with_loftr(f0, frames[j])
                t0 = type(pts0).__name__
                shape0 = getattr(pts0, "shape", None)
                log(
                    INFO,
                    f"[probe] LoFTR frame[0] vs frame[{j}]: "
                    f"return_type={t0} shape={shape0} n_matches={(0 if pts0 is None else len(pts0))}",
                )
            except Exception as e:
                log(WARNING, f"[probe] LoFTR test raised {type(e).__name__}: {e}")

    def _triangulate_pair(self, pts1, pts2, K, pose1, pose2, max_reproj_error):
        """
        Triangulate matched 2D points across two world-to-camera poses and return the
        surviving world-frame points together with the boolean mask of which input
        rows survived (cheirality + reprojection-error filter). Returns None if no
        points survive.
        """
        x1 = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, None).reshape(-1, 2).T.astype(np.float64)
        x2 = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, None).reshape(-1, 2).T.astype(np.float64)

        T21 = pose2 @ np.linalg.inv(pose1)
        R = T21[:3, :3]
        t = T21[:3, 3:4]

        P0 = np.hstack([np.eye(3), np.zeros((3, 1))])
        P1 = np.hstack([R, t])
        Xh = cv2.triangulatePoints(P0, P1, x1, x2)
        # Reject points at/near infinity before dividing — these would produce
        # inf/nan and pollute downstream projections.
        finite_h = np.abs(Xh[3]) > 1e-8
        with np.errstate(divide="ignore", invalid="ignore"):
            X = (Xh[:3] / Xh[3]).T
        finite_h = finite_h & np.isfinite(X).all(axis=1)
        if not finite_h.any():
            return None

        z1 = X[:, 2]
        with np.errstate(invalid="ignore", over="ignore"):
            X2 = (R @ X.T + t).T
            z2 = X2[:, 2]
            proj1 = (K @ X.T).T
            proj2 = (K @ X2.T).T
            proj1 = proj1[:, :2] / proj1[:, 2:3]
            proj2 = proj2[:, :2] / proj2[:, 2:3]

        # Mark non-finite rows as failures up front.
        finite_proj = (
            np.isfinite(proj1).all(axis=1) & np.isfinite(proj2).all(axis=1)
            & np.isfinite(z1) & np.isfinite(z2)
        )
        ok_input = finite_h & finite_proj

        mask = self._filter_triangulated_points(
            points_3d=X,
            pts1=pts1,
            pts2=pts2,
            proj1=proj1,
            proj2=proj2,
            z1=z1,
            z2=z2,
            max_reproj_error=max_reproj_error,
        )
        mask = mask & ok_input

        if not mask.any():
            return None

        Twc1 = np.linalg.inv(pose1)
        Xc1 = X[mask]
        Xh = np.hstack([Xc1, np.ones((Xc1.shape[0], 1))])
        Xw = (Twc1 @ Xh.T).T[:, :3]
        return Xw, mask

    def estimate_pose_from_matches(self, pts1, pts2, K):
        """
        Estimate the relative pose from the points
        :return: Essential (3x3) matrix, mask for inline/outline
        """
        E, mask = cv2.findEssentialMat(
            pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        if E is None or mask is None:
            return None, None, None

        mask = mask.ravel().astype(bool)
        if mask.sum() < 8:
            return None, None, None

        # homography-vs-essential degeneracy check on inliers
        H, mH = cv2.findHomography(
            pts1[mask], pts2[mask],
            cv2.RANSAC, 3.0, confidence=0.999
        )
        inl_F = int(mask.sum())
        inl_H = int(mH.ravel().sum()) if mH is not None else 0
        if inl_H >= 1.5 * inl_F:
            return None, None, None

        # Recover pose directly from E (essential matrix)
        ok, R, t, pose_mask = cv2.recoverPose(
            E, pts1, pts2, K, mask=mask.astype(np.uint8)
        )
        if not ok:
            return None, None, None

        inliers = pose_mask.ravel().astype(bool)
        return R, t, inliers

    def triangulate_points(self, pts1, pts2, K, pose1, pose2, max_reproj_error=2.0):
        """
        Triangulate matched 2D points across two world-to-camera poses and return
        the world-frame 3D points that pass cheirality and reprojection filters.
        """
        result = self._triangulate_pair(pts1, pts2, K, pose1, pose2, max_reproj_error)
        if result is None:
            return None
        return result[0]

    def _too_similar(self, img1, img2,
                     corr_t=0.995, mad_t=0.6, flow_px=0.6, max_corners=400):
        """
        Function to check if frames are too similar and skip - written by AI because I can't be bothered
        TODO: Improve this
        """
        # downscale + grayscale
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        h, w = g1.shape
        s = 256.0 / max(h, w)
        if s < 1.0:
            g1 = cv2.resize(g1, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            g2 = cv2.resize(g2, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

        # global histogram correlation (very cheap)
        hist1 = cv2.calcHist([g1], [0], None, [32], [0, 256]);
        cv2.normalize(hist1, hist1)
        hist2 = cv2.calcHist([g2], [0], None, [32], [0, 256]);
        cv2.normalize(hist2, hist2)
        corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        if corr >= corr_t:
            return True  # near-identical appearance

        # mean absolute difference after blur (robust to noise)
        b1 = cv2.GaussianBlur(g1, (3, 3), 0)
        b2 = cv2.GaussianBlur(g2, (3, 3), 0)
        mad = float(np.mean(np.abs(b1.astype(np.float32) - b2.astype(np.float32))))
        if mad <= mad_t:
            return True  # photometrically too close

        # tiny KLT probe to measure motion
        p0 = cv2.goodFeaturesToTrack(g1, maxCorners=max_corners, qualityLevel=0.01, minDistance=7,
                                     useHarrisDetector=True, k=0.04)
        if p0 is None or len(p0) < 20:
            return True  # no texture, treat as unusable
        p1, st, _ = cv2.calcOpticalFlowPyrLK(g1, g2, p0, None, winSize=(21, 21), maxLevel=2,
                                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-3))
        if p1 is None or st is None:
            return True
        st = st.reshape(-1).astype(bool)
        flow = np.linalg.norm(p1[st] - p0[st], axis=2).reshape(-1)
        return float(np.median(flow)) < flow_px

    def _filter_triangulated_points(
        self,
        points_3d,
        pts1,
        pts2,
        proj1,
        proj2,
        z1,
        z2,
        max_reproj_error=2.0,
    ):
        """
        Filters the 3D points based on reprojection error and depth
        :param points_3d: 3D points in cam1
        :param pts1: First 2D point Nx2 pixels
        :param pts2: Second 2D point Nx2 pixels
        :param proj1: Projection 1 Nx2 pixels predicted
        :param proj2: Projection 2 Nx2 pixels predicted
        :param z1: N depths in cam1
        :param z2: N depths in cam2
        :param max_reproj_error: Threshold for reproj error - Needs to be configurable
        :return: mask to filter points
        """
        # cheirality - no magnitude check
        cheirality = (z1 > 0) & (z2 > 0)

        # Pixel reprojection errors
        e1 = np.linalg.norm(proj1 - pts1, axis=1)
        e2 = np.linalg.norm(proj2 - pts2, axis=1)
        reproj_ok = (e1 < max_reproj_error) & (e2 < max_reproj_error)

        final_mask = cheirality & reproj_ok
        return final_mask


    def _extract_point_colors(self, frame, pts):
        """
        Extracts RGB colors at locations
        :param frame: Frame to get colors from
        :param pts: Points on the frame
        :return: Colors matrix
        """

        colors = []
        h, w = frame.shape[:2]
        for pt in pts:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < w and 0 <= y < h:
                color = frame[y, x] / 255.0  # Normalize
                colors.append(color)
            else:
                log(WARNING, "Error in extracting colors")
                colors.append([0.5, 0.5, 0.5])  # Default gray
        return colors
