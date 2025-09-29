import cv2
import torch
import numpy as np
from scipy.optimize import least_squares
from logging import log, INFO, WARNING
from torch.nn import functional as F


class Calibrator:
    def __init__(self, matcher="opencv"):
        """
        Initializes camera calibrator with matcher type
        :param matcher: Uses loftr if available, otherwise OpenCV
        """
        self.loftr = None
        self.matcher = None
        self.alg = None

        self.matcher = matcher
        self.setup_matcher()

    def setup_matcher(self):
        """
        Initializes feature matches.
        """
        try:
            import kornia.feature as KF

            self.matcher_type = "loftr"
            self.loftr = KF.LoFTR(pretrained="outdoor").eval()
            if torch.cuda.is_available():
                self.loftr = self.loftr.cuda()
        except ImportError:
            self.matcher_type = "opencv"
            self.alg = cv2.ORB.create(nfeatures=2000)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def extract_all_matches(self, frames):
        """
        Extracts all matches from video on GPU
        :param frames: all video frames
        :return: a np array of all matches in the form of (frame_idx1, frame_idx2, pts1, pts2)
        """
        log(INFO, f"Processing {len(frames)} frames")
        all_matches = []
        for i in range(len(frames) - 1):
            if i % 100 == 0:
                log(INFO, f"Matching frames {i} and {i + 1}")

            if self.matcher_type == "opencv":
                gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)

                # Detect + compute
                kp1, des1 = self.alg.detectAndCompute(gray1, None)
                kp2, des2 = self.alg.detectAndCompute(gray2, None)

                if not (des1 and des2):
                    log(WARNING, "Error in matching! - Skipping")
                    continue

                matches = self.matcher.knnMatch(des1, des2, k=2)

                # Lowe's ratio test
                # https://sites.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj2/html/sshah426/index.html
                # High ratio means clear match
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)

                if len(good_matches) < 20:
                    log(WARNING, f"Few good matches ({len(good_matches)}) - Skipping")
                    continue

                pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
                pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

            else:  # LoFTR matching
                pts1, pts2 = self.match_with_loftr(frames[i], frames[i + 1])

            all_matches.append(
                {"frame_i": i, "frame_j": i + 1, "pts1": pts1, "pts2": pts2}
            )
        return all_matches

    def match_with_loftr(self, img1, img2):
        """
        Uses LoFTR matching (much better than OpenCV) to get matches.
        Runs on GPU
        :param img1: First image
        :param img2: Second image
        :return: matching points
        """
        import kornia as K

        img1_t = K.image_to_tensor(img1).float() / 255.0  # normalize
        img2_t = K.image_to_tensor(img2).float() / 255.0

        if len(img1_t.shape) == 3:
            img1_t = img1_t.unsqueeze(0)
            img2_t = img2_t.unsqueeze(0)

        # Grayscale
        img1_gray = K.color.rgb_to_grayscale(img1_t)
        img2_gray = K.color.rgb_to_grayscale(img2_t)

        with torch.no_grad():
            input_dict = {"image0": img1_gray, "image1": img2_gray}
            correspondences = self.loftr(input_dict)

            mkpts0 = correspondences["keypoints0"].cpu().numpy()
            mkpts1 = correspondences["keypoints1"].cpu().numpy()
            confidence = correspondences["confidence"].cpu().numpy()

            # Filter by confidence
            mask = confidence > 0.5
            mkpts0 = mkpts0[mask]
            mkpts1 = mkpts1[mask]

        return mkpts0, mkpts1

    def refine_with_bundle_adjustment(self, matches, K_init):
        """
        Optimizes camera intrinsics and poses
        :param matches: Matches from video
        :param K_init: initial K guess
        :return: final K
        """
        point_3d = None
        observed_2d = None

        def objective(params):
            """
            Optimizes for focal length, poses, and 3D points
            :param params: focal, poses, points
            :return:
            """
            focal = params[0]
            K = np.array(
                [[focal, 0, K_init[0, 2]], [0, focal, K_init[1, 2]], [0, 0, 1]]
            )

            errors = []

            for match in matches:
                # Project from 3d to 2d
                proj = point_3d  # Dummy
                errors.append(proj - observed_2d)

            return np.concatenate(errors)

        initial_params = [K_init[0, 0]]

        # Optimize with least-squares for simplicity
        res = least_squares(
            objective,
            initial_params,
            method="trf",  # Trust Region Reflective - Stable
            verbose=2,
        )

        focal = res.x[0]
        return np.array([[focal, 0, K_init[0, 2]], [0, focal, K_init[1, 2]], [0, 0, 1]])

    def validate_intrinsics(self, K, matches):
        """
        Validates that intrinsics are accurate and checks error
        :param K: Intrinsics
        :param matches: Matches precomputed
        :return: mean_error
        """
        E = K.T @ F @ K

        errors = []
        for pt1, pt2 in matches:
            l = E @ np.append(pt1, 1)

            # Distance from pt2 to line
            err = abs(np.dot(l, np.append(pt2, 1))) / np.linalg.norm(l[:2])
            errors.append(err)

        return np.mean(errors)  # mean error

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

        # COLMAP initial guess formula - works surprisingly well
        initial_focal = 1.2 * max(width, height)

        # Principle points of camera
        cx, cy = width / 2, height / 2
        K_init = np.array(
            [[initial_focal, 0, cx], [0, initial_focal, cy], [0, 0, 1]]
        )  # https://ksimek.github.io/2013/08/13/intrinsic

        print(f"Initial guess: focal={initial_focal:.1f}")

        matches = self.extract_all_matches(video_path)
        print(f"Number of matches: {len(matches)}")

        # Refine
        K_refined = self.refine_with_bundle_adjustment(matches, K_init)

        error = self.validate_intrinsics(K_refined, matches)

        if error < 2.0:
            log(INFO, f"Calibration is good! Error: {error}")
        elif error < 5.0:
            log(WARNING, f"Calibration is okay, Error: {error}")
        else:
            log(WARNING, f"Poor calibration - could still work. Error: {error}")
        return K_refined
