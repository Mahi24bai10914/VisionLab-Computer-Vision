import cv2


def binary_threshold(gray_image, threshold=127):
    """
    Apply Binary Thresholding.
    """
    _, result = cv2.threshold(
        gray_image,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    return result


def otsu_threshold(gray_image):
    """
    Apply Otsu Thresholding.
    """
    _, result = cv2.threshold(
        gray_image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return result


def adaptive_threshold(gray_image):
    """
    Apply Adaptive Gaussian Thresholding.
    """
    return cv2.adaptiveThreshold(
        gray_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )