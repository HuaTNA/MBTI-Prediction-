"""
Face Detection Module
OpenCV DNN-based face detection (replaces MediaPipe for better compatibility)
"""

import os
import cv2
import logging

logger = logging.getLogger(__name__)


class FaceDetector:
    """Face detector using OpenCV DNN with Caffe SSD model"""

    def __init__(self, prototxt_path=None, model_path=None, confidence_threshold=0.5):
        """
        Initialize face detector

        Args:
            prototxt_path: Path to deploy.prototxt file
            model_path: Path to .caffemodel file
            confidence_threshold: Minimum confidence for face detection
        """
        self.confidence_threshold = confidence_threshold
        self.detector = None

        if prototxt_path and model_path:
            self.load_model(prototxt_path, model_path)

    def load_model(self, prototxt_path, model_path):
        """
        Load OpenCV DNN face detection model

        Args:
            prototxt_path: Path to deploy.prototxt file
            model_path: Path to .caffemodel file

        Raises:
            FileNotFoundError: If model files don't exist
        """
        # Check if model files exist
        if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Face detection model files not found. Please run: python download_face_models.py\n"
                f"Missing files:\n"
                f"  - {prototxt_path}\n"
                f"  - {model_path}"
            )

        # Load OpenCV DNN model
        self.detector = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        logger.info("Successfully initialized OpenCV DNN face detector (replaces MediaPipe)")

    def detect_faces(self, image):
        """
        Detect faces in image

        Args:
            image: Input image (BGR format)

        Returns:
            list: List of face bounding boxes as (x1, y1, x2, y2, confidence)
        """
        if self.detector is None:
            raise RuntimeError("Face detector not initialized. Call load_model() first.")

        h, w = image.shape[:2]

        # Preprocess image for DNN model
        # Create blob: resize to 300x300, subtract mean values
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        # Set input and perform forward pass
        self.detector.setInput(blob)
        detections = self.detector.forward()

        # Extract face detections
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.confidence_threshold:
                # Get bounding box coordinates
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                x1, y1, x2, y2 = box.astype("int")
                faces.append((x1, y1, x2, y2, float(confidence)))

        return faces

    def detect_and_crop_face(self, image, padding=0.2):
        """
        Detect and crop the most confident face from image

        Args:
            image: Input image (BGR format)
            padding: Extra padding around face as proportion of box size

        Returns:
            tuple: (cropped_face, face_detected)
                - cropped_face: Cropped face image or original image if no face detected
                - face_detected: Boolean indicating if face was found
        """
        faces = self.detect_faces(image)

        # If no faces detected, return original image
        if not faces:
            return image, False

        # Get the face with highest confidence
        best_face = max(faces, key=lambda f: f[4])
        x1, y1, x2, y2, confidence = best_face

        # Add padding
        h, w = image.shape[:2]
        box_width = x2 - x1
        box_height = y2 - y1
        x1 = max(0, int(x1 - box_width * padding))
        y1 = max(0, int(y1 - box_height * padding))
        x2 = min(w, int(x2 + box_width * padding))
        y2 = min(h, int(y2 + box_height * padding))

        # Crop face region
        face_crop = image[y1:y2, x1:x2]

        # Ensure cropped face is not empty
        if face_crop.size == 0:
            return image, False

        return face_crop, True


# Global face detector instance (singleton pattern)
_global_detector = None


def get_face_detector(prototxt_path=None, model_path=None):
    """
    Get global face detector instance (singleton)

    Args:
        prototxt_path: Path to deploy.prototxt (only needed on first call)
        model_path: Path to .caffemodel (only needed on first call)

    Returns:
        FaceDetector: Global face detector instance
    """
    global _global_detector

    if _global_detector is None:
        if prototxt_path is None or model_path is None:
            raise ValueError("Must provide model paths on first initialization")
        _global_detector = FaceDetector(prototxt_path, model_path)

    return _global_detector


def detect_and_crop_face(image, padding=0.2):
    """
    Convenience function to detect and crop face using global detector

    Args:
        image: Input image (BGR format)
        padding: Extra padding around face as proportion of box size

    Returns:
        tuple: (cropped_face, face_detected)
    """
    detector = get_face_detector()
    return detector.detect_and_crop_face(image, padding)
