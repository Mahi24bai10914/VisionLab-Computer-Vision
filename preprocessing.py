import cv2


def convert_to_grayscale(image):
    """
    Convert an image from BGR/RGB format to grayscale.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(gray_image):
    """
    Apply Gaussian Blur to reduce image noise.
    """
    return cv2.GaussianBlur(
        gray_image,
        (5, 5),
        0
    )


def preprocess_image(image):
    """
    Perform basic image preprocessing.
    """
    gray = convert_to_grayscale(image)
    blurred = apply_gaussian_blur(gray)

    return gray, blurred