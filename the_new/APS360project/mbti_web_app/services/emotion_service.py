"""
Emotion Analysis Service
Business logic for facial emotion recognition
"""

import cv2
import torch
import torch.nn.functional as F
import logging
from PIL import Image

import config
from services.model_loader import get_model_manager

logger = logging.getLogger(__name__)


def analyze_emotion(image):
    """
    Analyze facial emotion from image using CNN model

    Args:
        image: Input image (BGR format, numpy array)

    Returns:
        tuple: (emotion_dict, face_detected)
            - emotion_dict: Dictionary mapping emotion names to probabilities
            - face_detected: Boolean indicating if a face was found
    """
    manager = get_model_manager()

    try:
        # Detect and crop face
        face_image, face_detected = manager.face_detector.detect_and_crop_face(image)

        if not face_detected:
            logger.warning("No face detected, using full image for analysis")

        # Convert to RGB (OpenCV uses BGR)
        image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_img = Image.fromarray(image_rgb)

        # Apply transformation
        input_tensor = manager.emotion_transform(pil_img)
        input_tensor = input_tensor.unsqueeze(0).to(manager.device)

        # Inference
        with torch.no_grad():
            predictions, _ = manager.emotion_model(input_tensor)
            probabilities = F.softmax(predictions, dim=1)[0].cpu().numpy()

        # Create emotion dictionary
        emotion_dict = {
            emotion: float(prob)
            for emotion, prob in zip(config.EMOTION_CATEGORIES, probabilities)
        }

        return emotion_dict, face_detected

    except Exception as e:
        logger.error(f"Error in emotion analysis: {e}")
        # Return uniform probabilities as fallback
        fallback_prob = 1.0 / len(config.EMOTION_CATEGORIES)
        return {emotion: fallback_prob for emotion in config.EMOTION_CATEGORIES}, False
