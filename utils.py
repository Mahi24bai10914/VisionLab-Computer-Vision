from PIL import Image
import numpy as np
import cv2


def load_image(uploaded_file):
    """
    Load an uploaded image and convert it to RGB and BGR formats.
    """

    image = Image.open(uploaded_file).convert("RGB")

    image_rgb = np.array(image)

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )

    return image, image_rgb, image_bgr