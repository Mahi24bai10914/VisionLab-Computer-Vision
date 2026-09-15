import cv2


def histogram_equalization(gray_image):
    """
    Improve image contrast using Histogram Equalization.
    """
    return cv2.equalizeHist(gray_image)


def get_histogram(gray_image):
    """
    Calculate the intensity histogram of a grayscale image.
    """
    histogram = cv2.calcHist(
        [gray_image],
        [0],
        None,
        [256],
        [0, 256]
    )

    return histogram