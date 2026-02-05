"""
Image Processing Utilities
Functions for handling and processing image data
"""

import base64
import numpy as np
import cv2


def process_base64_image(base64_image):
    """
    Process base64 encoded image string

    Converts a base64 encoded image string to a numpy array
    suitable for processing with OpenCV

    Args:
        base64_image: Base64 encoded image string (may include data URL prefix)

    Returns:
        numpy.ndarray: Image as BGR numpy array (OpenCV format)
    """
    # Remove data URL prefix if present (e.g., "data:image/png;base64,...")
    if ',' in base64_image:
        base64_image = base64_image.split(',')[1]

    # Decode base64 image
    image_data = base64.b64decode(base64_image)

    # Convert to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return img
