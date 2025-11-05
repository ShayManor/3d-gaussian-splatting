# Implementation of RANSAC for matching videos
# Source: https://en.wikipedia.org/wiki/Random_sample_consensus
# Written by Claude for testing, will re-implement later
from logging import INFO, log
from typing import Dict

import numpy as np
from torch import svd
from scipy.spatial import KDTree
import cv2

from src.video.calibrate import Calibrator
from src.video.video_loader import VideoLoader


def _align_video_to_reference(video_data: Dict, reference: Dict) -> Dict:
    """
    Align a video to the reference coordinate system using robust feature matching and RANSAC
    """
    # If no points, return as is
    if len(video_data['points_3d']) == 0 or len(reference['points_3d']) == 0:
        return video_data

    # Step 1: Find overlapping frames between videos using image features
    print(f"Aligning video with {len(video_data['points_3d'])} points to reference...")

    # Load sample frames from both videos for feature matching
    ref_video_path = reference['video_info'][0]['path']
    new_video_path = video_data['video_path']

    ref_loader = VideoLoader(ref_video_path, cache_frames=False)
    new_loader = VideoLoader(new_video_path, cache_frames=False)

    # Sample frames evenly through both videos
    n_sample_frames = min(20, ref_loader.total_frames, new_loader.total_frames)
    ref_frame_indices = np.linspace(0, ref_loader.total_frames - 1, n_sample_frames).astype(int)
    new_frame_indices = np.linspace(0, new_loader.total_frames - 1, n_sample_frames).astype(int)

    # Find matching frame pairs between videos
    calibrator = Calibrator(matcher="loftr")  # Reuse existing matcher
    best_matches = []
    best_frame_pairs = []

    for ref_idx in ref_frame_indices[:10]:  # Limit to avoid too many comparisons
        ref_frame = ref_loader.get_frame(ref_idx)
        if ref_frame is None:
            continue

        for new_idx in new_frame_indices[:10]:
            new_frame = new_loader.get_frame(new_idx)
            if new_frame is None:
                continue

            # Match features between frames
            if calibrator.matcher_type == "loftr":
                pts1, pts2 = calibrator.match_with_loftr(ref_frame, new_frame)
            else:
                # Use OpenCV matching
                gray1 = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

                kp1, des1 = calibrator.alg.detectAndCompute(gray1, None)
                kp2, des2 = calibrator.alg.detectAndCompute(gray2, None)

                if des1 is None or des2 is None:
                    continue

                matches = calibrator.matcher.knnMatch(des1, des2, k=2)

                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)

                if len(good_matches) < 30:
                    continue

                pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
                pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

            if len(pts1) > 30:
                best_matches.append((pts1, pts2))
                best_frame_pairs.append((ref_idx, new_idx))

    if not best_matches:
        print("Warning: No overlapping views found, using fallback alignment")
        return _fallback_alignment(video_data, reference)

    # Step 2: Triangulate matched 2D points to get 3D correspondences
    ref_poses = reference['all_poses'][0]
    new_poses = video_data['poses']
    ref_K = reference['all_intrinsics'][0]
    new_K = video_data['intrinsics']

    correspondence_3d_ref = []
    correspondence_3d_new = []

    for (pts1, pts2), (ref_frame_idx, new_frame_idx) in zip(best_matches, best_frame_pairs):
        # Get closest poses for these frame indices
        ref_pose_idx = min(ref_frame_idx // 10, len(ref_poses) - 1)
        new_pose_idx = min(new_frame_idx // 10, len(new_poses) - 1)

        ref_pose = ref_poses[ref_pose_idx]
        new_pose = new_poses[new_pose_idx]

        # Triangulate points in reference coordinate system
        P1_ref = ref_K @ ref_pose[:3, :]
        P2_ref = ref_K @ ref_poses[min(ref_pose_idx + 1, len(ref_poses) - 1)][:3, :]

        pts1_h = cv2.undistortPoints(pts1.reshape(-1, 1, 2), ref_K, None)
        pts2_h = cv2.undistortPoints(pts2.reshape(-1, 1, 2), new_K, None)

        # Triangulate in reference space
        points_4d = cv2.triangulatePoints(P1_ref, P2_ref, pts1_h, pts2_h)  # Not sure if pts1_h, pts2_h or both pts1
        points_3d_ref = (points_4d[:3] / points_4d[3]).T

        # Triangulate same points in new video space
        P1_new = new_K @ new_pose[:3, :]
        P2_new = new_K @ new_poses[min(new_pose_idx + 1, len(new_poses) - 1)][:3, :]

        points_4d = cv2.triangulatePoints(P1_new, P2_new, pts1_h, pts2_h)
        points_3d_new = (points_4d[:3] / points_4d[3]).T

        # Filter outliers based on reasonable depths
        valid_mask = (np.abs(points_3d_ref[:, 2]) < 100) & \
                     (np.abs(points_3d_ref[:, 2]) > 0.1) & \
                     (np.abs(points_3d_new[:, 2]) < 100) & \
                     (np.abs(points_3d_new[:, 2]) > 0.1)

        correspondence_3d_ref.extend(points_3d_ref[valid_mask])
        correspondence_3d_new.extend(points_3d_new[valid_mask])

    if len(correspondence_3d_ref) < 10:
        print("Warning: Too few 3D correspondences, using fallback alignment")
        return _fallback_alignment(video_data, reference)

    correspondence_3d_ref = np.array(correspondence_3d_ref)
    correspondence_3d_new = np.array(correspondence_3d_new)

    # Step 3: Find transformation using RANSAC
    best_transform = _ransac_rigid_transform(
        correspondence_3d_new,
        correspondence_3d_ref,
        max_iterations=1000,
        threshold=0.5
    )

    if best_transform is None:
        print("Warning: RANSAC failed, using fallback alignment")
        return _fallback_alignment(video_data, reference)

    # Step 4: Apply transformation to all points and poses
    R, t, s = best_transform

    # Transform points
    aligned_points = s * (video_data['points_3d'] @ R.T) + t

    # Transform poses
    aligned_poses = []
    for pose in video_data['poses']:
        aligned_pose = np.eye(4)
        aligned_pose[:3, :3] = pose[:3, :3] @ R.T
        aligned_pose[:3, 3] = s * (pose[:3, 3] @ R.T) + t
        aligned_poses.append(aligned_pose)

    # Step 5: Refine with ICP if we have enough overlapping points
    if len(correspondence_3d_ref) > 100:
        refined_transform = _icp_refinement(
            aligned_points,
            reference['points_3d'],
            max_iterations=50
        )

        if refined_transform is not None:
            R_icp, t_icp = refined_transform
            aligned_points = (aligned_points @ R_icp.T) + t_icp

            for i, pose in enumerate(aligned_poses):
                refined_pose = np.eye(4)
                refined_pose[:3, :3] = pose[:3, :3] @ R_icp.T
                refined_pose[:3, 3] = (pose[:3, 3] @ R_icp.T) + t_icp
                aligned_poses[i] = refined_pose

    log(INFO, f"Alignment complete - scale: {s:.3f}, translation: {t}")
    log(INFO,f"Check RGB (should be in range 0-255), Max Color: {video_data['colors'].max()}")

    return {
        'points_3d': aligned_points,
        'colors': video_data['colors'],
        'poses': np.array(aligned_poses)
    }


def _ransac_rigid_transform(source_points, target_points,
                            max_iterations=1000, threshold=0.5):
    """
    RANSAC-based estimation of rigid transformation with scale
    Returns: (R, t, s) - rotation, translation, scale
    """
    n_points = len(source_points)
    if n_points < 4:
        return None

    best_inliers = 0
    best_transform = None

    for _ in range(max_iterations):
        # Sample 4 random correspondences
        idx = np.random.choice(n_points, min(4, n_points), replace=False)
        src_sample = source_points[idx]
        tgt_sample = target_points[idx]

        # Estimate transformation from samples
        transform = _estimate_similarity_transform(src_sample, tgt_sample)
        if transform is None:
            continue

        R, t, s = transform

        # Transform all source points
        transformed = s * (source_points @ R.T) + t

        # Count inliers
        distances = np.linalg.norm(transformed - target_points, axis=1)
        inliers = distances < threshold
        n_inliers = inliers.sum()

        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_transform = transform

            # Early termination if we have enough inliers
            if n_inliers > 0.8 * n_points:
                break

    # Refine using all inliers
    if best_transform is not None and best_inliers > 10:
        R, t, s = best_transform
        transformed = s * (source_points @ R.T) + t
        distances = np.linalg.norm(transformed - target_points, axis=1)
        inliers = distances < threshold

        if inliers.sum() > 4:
            best_transform = _estimate_similarity_transform(
                source_points[inliers],
                target_points[inliers]
            )

    return best_transform


def _estimate_similarity_transform(source, target):
    """
    Estimate similarity transformation (rotation + translation + uniform scale)
    Using Umeyama's method
    """
    # Center the points
    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)

    src_centered = source - src_mean
    tgt_centered = target - tgt_mean

    # Compute scale
    src_var = np.sum(src_centered ** 2) / len(source)
    tgt_var = np.sum(tgt_centered ** 2) / len(target)

    if src_var < 1e-10:
        return None

    # Compute rotation using SVD
    H = src_centered.T @ tgt_centered
    U, S, Vt = svd(H)
    R = Vt.T @ U.T

    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute scale
    scale = np.sqrt(tgt_var / src_var)

    # Compute translation
    t = tgt_mean - scale * (src_mean @ R.T)

    return R, t, scale


def _icp_refinement(source_points, target_points, max_iterations=50):
    """
    ICP refinement for better alignment
    """


    current_source = source_points.copy()
    total_R = np.eye(3)
    total_t = np.zeros(3)

    # Build KD-tree for target
    target_tree = KDTree(target_points)

    prev_error = float('inf')

    for iteration in range(max_iterations):
        # Find nearest neighbors
        distances, indices = target_tree.query(current_source, k=1)

        # Filter outliers
        median_dist = np.median(distances)
        mask = distances < 3 * median_dist

        if mask.sum() < 10:
            break

        # Estimate transformation
        matched_target = target_points[indices[mask]]
        transform = _estimate_rigid_transform(
            current_source[mask],
            matched_target
        )

        if transform is None:
            break

        R, t = transform

        # Apply transformation
        current_source = (current_source @ R.T) + t
        total_R = R @ total_R
        total_t = (total_t @ R.T) + t

        # Check convergence
        error = np.mean(distances[mask])
        if abs(error - prev_error) < 1e-6:
            break
        prev_error = error

    return total_R, total_t


def _estimate_rigid_transform(source, target):
    """
    Estimate rigid transformation (rotation + translation only, no scale)
    """
    # Center the points
    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)

    src_centered = source - src_mean
    tgt_centered = target - tgt_mean

    # Compute rotation using SVD
    H = src_centered.T @ tgt_centered
    U, S, Vt = svd(H)
    R = Vt.T @ U.T

    # Ensure proper rotation
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation
    t = tgt_mean - (src_mean @ R.T)

    return R, t


def _fallback_alignment(video_data: Dict, reference: Dict) -> Dict:
    """
    Fallback to simpler alignment when feature matching fails
    """
    if len(video_data['points_3d']) == 0:
        return video_data

    # Use PCA-based alignment as fallback
    from sklearn.decomposition import PCA

    # Fit PCA on both point clouds
    pca_ref = PCA(n_components=3)
    pca_new = PCA(n_components=3)

    pca_ref.fit(reference['points_3d'])
    pca_new.fit(video_data['points_3d'])

    # Align principal components
    R = pca_ref.components_.T @ pca_new.components_

    # Align centroids
    ref_center = reference['points_3d'].mean(axis=0)
    new_center = video_data['points_3d'].mean(axis=0)
    t = ref_center - (new_center @ R.T)

    # Apply transformation
    aligned_points = (video_data['points_3d'] @ R.T) + t

    aligned_poses = []
    for pose in video_data['poses']:
        aligned_pose = np.eye(4)
        aligned_pose[:3, :3] = pose[:3, :3] @ R.T
        aligned_pose[:3, 3] = (pose[:3, 3] @ R.T) + t
        aligned_poses.append(aligned_pose)

    return {
        'points_3d': aligned_points,
        'colors': video_data['colors'],
        'poses': np.array(aligned_poses)
    }