import cv2
import torch
import numpy as np
from scipy.optimize import least_squares
from logging import log, INFO, WARNING


class Calibrator:
    def __init__(self, matcher="sift", focal_px=None, focal_35mm=None):
        """
        Initializes camera calibrator with matcher type.

        focal_px overrides the heuristic `1.2 * max(W, H)`. focal_35mm gives the
        same override expressed in 35mm-equivalent millimeters (e.g. iPhone 17
        main camera = 24); converted via f_px = (focal_35mm / 36) * max(W, H).
        Pass at most one. None on both falls back to the heuristic.
        """
        if focal_px is not None and focal_35mm is not None:
            raise ValueError("Pass focal_px OR focal_35mm, not both")
        self.focal_px = float(focal_px) if focal_px is not None else None
        self.focal_35mm = float(focal_35mm) if focal_35mm is not None else None

        self.loftr = None
        self.matcher = None
        self.alg = None

        self.matcher_type = matcher
        self.setup_matcher()

    def setup_matcher(self):
        """
        Initialize the requested feature matcher.

        Honors `self.matcher_type` set in __init__:
          - "sift"   : OpenCV SIFT (corner-based, no GPU) — RECOMMENDED.
          - "loftr"  : kornia LoFTR (dense grid). Falls back to SIFT on ImportError.
          - "opencv" : ORB descriptor + Lowe ratio. Weak on tiny images; kept for tests.
        """
        if self.matcher_type == "loftr":
            try:
                import kornia.feature as KF

                model = KF.LoFTR(pretrained="outdoor")
                model.eval()
                self.loftr = model
                if torch.cuda.is_available():
                    self.loftr = self.loftr.cuda()
                return
            except ImportError:
                log(WARNING, "LoFTR requested but kornia is unavailable; falling back to SIFT")
                self.matcher_type = "sift"

        if self.matcher_type == "sift":
            self.alg = cv2.SIFT.create(nfeatures=4000, contrastThreshold=0.005, edgeThreshold=20)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            return

        # ORB fallback (matcher_type == "opencv")
        self.alg = cv2.ORB.create(nfeatures=3000, scoreType=cv2.ORB_HARRIS_SCORE, fastThreshold=20)
        # crossCheck must stay off — match_with_opencv runs Lowe's ratio test which
        # needs knnMatch(k=2), and OpenCV refuses k>1 when crossCheck is enabled.
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def extract_all_matches(self, frames):
        """
        Extracts all matches from video on GPU
        :param frames: all video frames
        :return: a np array of all matches in the form of (frame_idx1, frame_idx2, pts1, pts2)
        """
        all_matches = []
        for i in range(len(frames) - 1):
            if self.matcher_type in ("opencv", "sift"):
                pts1, pts2 = self.match_with_opencv(frames[i], frames[i + 1])
            else:
                pts1, pts2 = self.match_with_loftr(frames[i], frames[i + 1])

            if any((obj is None) or (not isinstance(obj, np.ndarray)) for obj in (pts1, pts2)):
                continue

            all_matches.append(
                {"frame_i": i, "frame_j": i + 1, "pts1": pts1, "pts2": pts2}
            )
        return all_matches

    def match_with_opencv(self, frame1, frame2, threshold=None, num_matches=None):
        """
        Detect + describe via self.alg, knn-match via self.matcher, run Lowe's
        ratio test, return matched pixel coords. SIFT and ORB defaults differ:
        SIFT produces fewer descriptors than ORB on small images, so the gates
        loosen slightly when the active descriptor is SIFT.
        """
        if threshold is None:
            threshold = 0.75 if self.matcher_type == "sift" else 0.7
        if num_matches is None:
            num_matches = 8 if self.matcher_type == "sift" else 20

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        kp1, des1 = self.alg.detectAndCompute(gray1, None)
        kp2, des2 = self.alg.detectAndCompute(gray2, None)

        if any((obj is None) or (len(obj) == 0) for obj in (des1, des2)):
            log(WARNING, "Error in matching! - Skipping")
            return None, None

        matches = self.matcher.knnMatch(des1, des2, k=2)

        # Lowe's ratio test — only keep matches where the best is clearly better
        # than the runner-up.
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < threshold * n.distance:
                    good_matches.append(m)

        if len(good_matches) < num_matches:
            log(WARNING, f"Few good matches ({len(good_matches)}) - Skipping")
            return None, None

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        return pts1, pts2

    def match_with_loftr(self, img1, img2):
        """
        Run kornia LoFTR on a pair and return matched pixel coords as **numpy
        ndarrays**. Always returns numpy regardless of device — the downstream
        SfM code (`extract_all_matches`) filters with `isinstance(.., np.ndarray)`
        so torch tensors get silently dropped.
        """
        import kornia as K

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img1_t = K.image_to_tensor(img1).float() / 255.0
        img2_t = K.image_to_tensor(img2).float() / 255.0

        if img1_t.dim() == 3:
            img1_t = img1_t.unsqueeze(0)
            img2_t = img2_t.unsqueeze(0)

        img1_gray = K.color.rgb_to_grayscale(img1_t)
        img2_gray = K.color.rgb_to_grayscale(img2_t)

        device = next(self.loftr.parameters()).device
        img1_gray = img1_gray.to(device)
        img2_gray = img2_gray.to(device)

        with torch.no_grad():
            correspondences = self.loftr({"image0": img1_gray, "image1": img2_gray})

        mkpts0 = correspondences["keypoints0"].detach().cpu().numpy()
        mkpts1 = correspondences["keypoints1"].detach().cpu().numpy()
        confidence = correspondences["confidence"].detach().cpu().numpy()

        mask = confidence > 0.5
        return mkpts0[mask], mkpts1[mask]

    def refine_with_bundle_adjustment(self, matches, K_init):
        """
        Optimizes camera intrinsics and poses
        :param matches: Matches from video
        :param K_init: initial K guess
        :return: final K
        """
        # Collect 3D-2D correspondences from matches
        points_3d_list = []
        points_2d_view1_list = []
        points_2d_view2_list = []
        poses = []

        for match in matches[:10]:
            pts1 = match["pts1"]
            pts2 = match["pts2"]

            if len(pts1) < 8:
                continue

            # Estimate relative pose
            E, mask = cv2.findEssentialMat(pts1, pts2, K_init, method=cv2.RANSAC)
            if E is None:
                continue

            _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K_init, mask=mask)

            # Triangulate points
            P1 = K_init @ np.hstack([np.eye(3), np.zeros((3, 1))])  # Identity pose
            P2 = K_init @ np.hstack([R, t])  # Relative pose

            pts1_h = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K_init, None)
            pts2_h = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K_init, None)

            points_4d = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
            points_3d = (points_4d[:3] / points_4d[3]).T

            # Filter valid points
            valid = (points_3d[:, 2] > 0.1) & (points_3d[:, 2] < 100)
            if valid.sum() >= 5:
                points_3d_list.append(points_3d[valid])
                points_2d_view1_list.append(pts1[valid])
                points_2d_view2_list.append(pts2[valid])
                poses.append((np.eye(3, 4), np.hstack([R, t])))

        if len(points_3d_list) == 0:
            log(WARNING, "No valid 3D points for bundle adjustment")
            return K_init

        def objective(params):
            """
            Optimize focal length by minimizing reprojection error
            :param params: [focal_length, cx_offset, cy_offset]
            """
            focal = params[0]

            K = np.array(
                [[focal, 0, K_init[0, 2]], [0, focal, K_init[1, 2]], [0, 0, 1]]
            )

            errors = []
            for i in range(len(points_3d_list)):
                pts1 = points_2d_view1_list[i]
                pts2 = points_2d_view2_list[i]
                pts_3d = points_3d_list[i]
                pose1, pose2 = poses[i]

                # View 1: Identity pose (world frame)
                P1 = K @ pose1
                P2 = K @ pose2

                # Project 3D points to 2D in both views
                # 3D->2D: p_2d = K @ [R|t] @ [X, Y, Z, 1]^T
                pts_3d_h = np.hstack([pts_3d, np.ones((len(pts_3d), 1))])  # Homogeneous

                # View 1 projection
                proj_v1 = (P1 @ pts_3d_h.T).T  # [N, 3]
                proj_v1 = proj_v1[:, :2] / proj_v1[:, 2:3]  # Normalize by Z

                # View 2 projection
                proj_v2 = (P2 @ pts_3d_h.T).T
                proj_v2 = proj_v2[:, :2] / proj_v2[:, 2:3]

                # Reprojection errors
                error_v1 = np.linalg.norm(proj_v1 - pts1, axis=1)
                error_v2 = np.linalg.norm(proj_v2 - pts2, axis=1)

                errors.extend(error_v1)
                errors.extend(error_v2)

            return np.array(errors) if len(errors) > 0 else np.array([10000.0])

        # Optimize: [focal, cx_offset, cy_offset]
        initial_params = [K_init[0, 0]]

        result = least_squares(
            objective,
            initial_params,
            method="trf",
            verbose=1,
            bounds=(
                [K_init[0, 0] * 0.5],
                [K_init[0, 0] * 2.0],
            ),  # Bound focal to within 50% of initial
            max_nfev=30,
            ftol=1e-4,
        )

        # Build refined K
        focal_refined = result.x[0]

        K_refined = np.array(
            [
                [focal_refined, 0, K_init[0, 2]],
                [0, focal_refined, K_init[1, 2]],
                [0, 0, 1],
            ]
        )

        log(INFO, f"Refined focal: {K_init[0, 0]:.1f} => {focal_refined:.1f}")

        return K_refined

    def validate_intrinsics(self, K, matches):
        """
        Reports mean Sampson distance (pixel-space symmetric epipolar error) on
        RANSAC inliers. Independent of K — F is computed from pixel matches and
        the line `l = F @ pt1_h` lives in pixel coords. Earlier versions used
        E = K^T F K to build the line, which conflates pixel and normalized
        coords and produces a K-dependent number that scales with the focal
        length (so a too-large focal heuristic inflated this metric).

        Threshold rule of thumb: <1 px clean, 1-3 px acceptable, >3 px implies
        a matching or geometry problem (NOT bad K — this metric doesn't depend
        on K).
        """
        if not matches:
            log(WARNING, "validate_intrinsics: no matches; skipping (returns inf)")
            return float("inf")
        errors = []
        n_sample = min(5, len(matches))
        sampled_matches = np.random.choice(matches, n_sample, replace=False)
        for match in sampled_matches:
            pts1 = match["pts1"]
            pts2 = match["pts2"]

            if len(pts1) < 8:
                log(WARNING, "Small points")
                continue

            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
            if F is None:
                log(WARNING, "Fundamental Matrix is None")
                continue

            for i in range(len(pts1)):
                if not mask[i]:
                    continue
                pt1_h = np.append(pts1[i], 1.0)
                pt2_h = np.append(pts2[i], 1.0)

                # Symmetric epipolar (Sampson) distance: average pixel-space
                # distance from each point to the epipolar line induced by
                # its match in the other image.
                l2 = F @ pt1_h          # epipolar line in image 2
                l1 = F.T @ pt2_h        # epipolar line in image 1
                d2 = abs(float(pt2_h @ l2)) / (np.linalg.norm(l2[:2]) + 1e-8)
                d1 = abs(float(pt1_h @ l1)) / (np.linalg.norm(l1[:2]) + 1e-8)
                errors.append(0.5 * (d1 + d2))
        return float(np.mean(errors)) if errors else float("inf")

    def identify_intrinsics(self, frames, video_path):
        """
        Estimates camera intrinsic matrix (focal length and principle) from  frame
        :param video_path: path to video
        :param frames: initial frame to get details
        :return: camera intrinsics matrix
        """
        import cv2
        import numpy as np

        capture = cv2.VideoCapture(video_path)
        width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Heuristic 1.2*max assumes ~45° FOV and is way off for phones (iPhone
        # main camera at 24mm-equiv has ~74° FOV → true f ≈ 0.67*max). Allow
        # explicit override via focal_px or focal_35mm.
        if self.focal_px is not None:
            initial_focal = self.focal_px
            source = "user override (focal_px)"
        elif self.focal_35mm is not None:
            initial_focal = (self.focal_35mm / 36.0) * max(width, height)
            source = f"user override (focal_35mm={self.focal_35mm})"
        else:
            initial_focal = 1.2 * max(width, height)
            source = "heuristic 1.2*max(W,H)"

        cx, cy = width / 2, height / 2
        K_init = np.array(
            [[initial_focal, 0, cx], [0, initial_focal, cy], [0, 0, 1]]
        )

        log(INFO, f"Focal: {initial_focal:.1f} px [{source}] (frame {int(width)}x{int(height)})")

        matches = self.extract_all_matches(frames)
        log(INFO, f"Number of matches: {len(matches)}")

        # Refine
        # K_refined = self.refine_with_bundle_adjustment(matches, K_init)
        K_refined = K_init

        error = self.validate_intrinsics(K_refined, matches)

        if error < 2.0:
            log(INFO, f"Calibration is good! Error: {error}")
        elif error < 5.0:
            log(WARNING, f"Calibration is okay, Error: {error}")
        else:
            log(WARNING, f"Poor calibration - could still work. Error: {error}")
        return K_refined

    def identify_intrinsics_cheap(self, image_path):
        """
        Estimates camera intrinsic matrix from image path quickly - for tests
        :param image_path: path to video
        :return: camera intrinsics matrix
        """
        import cv2
        import numpy as np

        img = cv2.imread(image_path)
        height, width = img.shape[:2]

        initial_focal = 1.2 * max(width, height)
        log(INFO, f"Focal: {initial_focal}")
        cx, cy = width / 2, height / 2
        K_init = np.array(
            [[initial_focal, 0, cx], [0, initial_focal, cy], [0, 0, 1]]
        )  # https://ksimek.github.io/2013/08/13/intrinsic
        return K_init
