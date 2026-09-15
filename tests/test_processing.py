import sys
import os

# Add the main project folder to Python's path
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import numpy as np
import cv2

from preprocessing import preprocess_image
from enhancement import histogram_equalization
from feature_extraction import detect_canny_edges
from segmentation import (
    binary_threshold,
    otsu_threshold,
    adaptive_threshold
)


def create_test_image():
    """
    Create a simple test image.
    """
    image = np.zeros(
        (200, 200, 3),
        dtype=np.uint8
    )

    cv2.rectangle(
        image,
        (50, 50),
        (150, 150),
        (255, 255, 255),
        -1
    )

    return image


def test_preprocessing():
    image = create_test_image()

    gray, blurred = preprocess_image(image)

    assert gray.shape == (200, 200)
    assert blurred.shape == (200, 200)


def test_histogram_equalization():
    image = create_test_image()

    gray, _ = preprocess_image(image)

    result = histogram_equalization(gray)

    assert result.shape == gray.shape


def test_canny():
    image = create_test_image()

    gray, blurred = preprocess_image(image)

    result = detect_canny_edges(blurred)

    assert result.shape == gray.shape


def test_segmentation():
    image = create_test_image()

    gray, _ = preprocess_image(image)

    binary = binary_threshold(gray)
    otsu = otsu_threshold(gray)
    adaptive = adaptive_threshold(gray)

    assert binary.shape == gray.shape
    assert otsu.shape == gray.shape
    assert adaptive.shape == gray.shape