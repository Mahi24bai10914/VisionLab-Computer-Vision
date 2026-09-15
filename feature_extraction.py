import cv2
import numpy as np


def detect_canny_edges(gray_image):
    """
    Detect edges using the Canny edge detection algorithm.
    """
    return cv2.Canny(
        gray_image,
        100,
        200
    )


def detect_harris_corners(gray_image, original_image):
    """
    Detect corners using the Harris Corner Detection algorithm.
    """

    gray_float = np.float32(gray_image)

    harris = cv2.cornerHarris(
        gray_float,
        2,
        3,
        0.04
    )

    harris = cv2.dilate(
        harris,
        None
    )

    result = original_image.copy()

    result[
        harris > 0.01 * harris.max()
    ] = [255, 0, 0]

    return result


def create_sift_detector():
    """
    Create a SIFT feature detector.
    """
    return cv2.SIFT_create()


def detect_sift_features(sift, gray_image):
    """
    Detect SIFT keypoints and descriptors.
    """
    keypoints, descriptors = sift.detectAndCompute(
        gray_image,
        None
    )

    return keypoints, descriptors
