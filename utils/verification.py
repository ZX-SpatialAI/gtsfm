"""Utilities for verification stage of the frontend.

Authors: Ayush Baid
"""
import cv2 as cv
import numpy as np
from gtsam import Cal3Bundler, EssentialMatrix, Point3, Rot3


def cast_essential_matrix_to_gtsam(im2_E_im1: np.ndarray,
                                   verified_keypoints_im1: np.ndarray,
                                   verified_keypoints_im2: np.ndarray,
                                   camera_intrinsics_im1: Cal3Bundler,
                                   camera_intrinsics_im2: Cal3Bundler
                                   ) -> EssentialMatrix:
    """Cast essential matrix from numpy matrix to gtsam type.

    Args:
        im2_E_im1: essential matrix as numpy matrix of shape 3x3.
        verified_keypoints_im1: keypoints from image #1 which form verified
                                correspondences, of shape (N, 2+).
        verified_keypoints_im2: keypoints from image #1 which form verified
                                correspondences, of shape (N, 2+).
        camera_intrinsics_im1: intrinsics for image #1.
        camera_intrinsics_im2: intrinsics for image #2.

    Returns:
        EssentialMatrix: [description]
    """
    # TODO(ayush): move it to GTSAM as a constructor.

    # obtain points in normalized coordinates using intrinsics.
    normalized_verified_keypoints_im1 = camera_intrinsics_im1.calibrate(
        verified_keypoints_im1[:, :2])
    normalized_verified_keypoints_im2 = camera_intrinsics_im2.calibrate(
        verified_keypoints_im2[:, :2])

    # use opencv to recover pose
    _, R, t, _ = cv.recoverPose(im2_E_im1,
                                normalized_verified_keypoints_im1,
                                normalized_verified_keypoints_im2)

    return EssentialMatrix(Rot3(R), Point3(t))


def fundamental_matrix_to_essential_matrix(im2_F_im1: np.ndarray,
                                           camera_intrinsics_im1: Cal3Bundler,
                                           camera_intrinsics_im2: Cal3Bundler
                                           ) -> np.ndarray:
    """Converts the fundamental matrix to essential matrix using camera
    intrinsics.

    Args:
        im2_F_im1: fundamental matrix which maps points in image #1 to lines in 
                   image #2. 
        camera_intrinsics_im1: intrinsics for image #1.
        camera_intrinsics_im2: intrinsics for image #2.

    Returns:
            Estimated essential matrix im2_E_im1.
    """
    return camera_intrinsics_im2.T @ im2_F_im1 @ camera_intrinsics_im1
