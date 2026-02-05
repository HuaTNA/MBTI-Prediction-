"""
Configuration constants for MBTI Assessment System
Contains all file paths, model settings, and system constants
"""

import os

# Base directory for models
BASE_DIR = r"E:\APS360_project\MBTI-Prediction-\the_new\APS360project\mbti_web_app\models"

# Emotion recognition model paths (PyTorch)
EMOTION_MODEL_PATH = os.path.join(BASE_DIR, "emotion", "improved_emotion_model.pth")

# Text MBTI model paths (Scikit-learn)
TEXT_MODEL_DIR = os.path.join(BASE_DIR, "text", "ml")
TEXT_MODEL_PATH = os.path.join(TEXT_MODEL_DIR, "model.pkl")
TEXT_VECTORIZER_PATH = os.path.join(TEXT_MODEL_DIR, "vectorizer.pkl")
TEXT_LABEL_ENCODER_PATH = os.path.join(TEXT_MODEL_DIR, "label_encoder.pkl")
TEXT_CONFIG_PATH = os.path.join(TEXT_MODEL_DIR, "config.json")

# Face detection model paths (OpenCV DNN)
FACE_DETECTION_MODEL_DIR = os.path.join(BASE_DIR, "face_detection")
FACE_DETECTION_PROTOTXT = os.path.join(FACE_DETECTION_MODEL_DIR, "deploy.prototxt")
FACE_DETECTION_MODEL = os.path.join(FACE_DETECTION_MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

# Question and description data directory
QUESTIONS_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
QUESTIONS_EN_FILE = os.path.join(QUESTIONS_DATA_DIR, 'questions_en.json')
QUESTIONS_ZH_FILE = os.path.join(QUESTIONS_DATA_DIR, 'questions_zh.json')
MBTI_DESCRIPTIONS_EN_FILE = os.path.join(QUESTIONS_DATA_DIR, 'mbti_descriptions_en.json')
MBTI_DESCRIPTIONS_ZH_FILE = os.path.join(QUESTIONS_DATA_DIR, 'mbti_descriptions_zh.json')

# Results directory for saving assessment outputs
RESULTS_DIR = os.path.join(os.path.expanduser("~"), "MBTI_Results")

# Emotion categories recognized by the model
EMOTION_CATEGORIES = [
    'Anger', 'Confusion', 'Contempt', 'Disgust',
    'Happiness', 'Neutral', 'Sadness', 'Surprise'
]

# MBTI dimensions
MBTI_DIMENSIONS = ['I/E', 'S/N', 'T/F', 'J/P']

# All 16 MBTI types
MBTI_TYPES = [
    'INTJ', 'INTP', 'ENTJ', 'ENTP',
    'INFJ', 'INFP', 'ENFJ', 'ENFP',
    'ISTJ', 'ISFJ', 'ESTJ', 'ESFJ',
    'ISTP', 'ISFP', 'ESTP', 'ESFP'
]

# Face detection confidence threshold
FACE_DETECTION_CONFIDENCE = 0.5

# Face crop padding (as proportion of bounding box size)
FACE_CROP_PADDING = 0.2

# Model parameters
EMOTION_MODEL_BACKBONE = 'efficientnet_b3'
EMOTION_MODEL_DROPOUT_RATES = [0.5, 0.4, 0.3]
EMOTION_MODEL_NUM_CLASSES = len(EMOTION_CATEGORIES)

# Image transformation parameters
IMAGE_SIZE = (224, 224)
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

# Flask server configuration
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = True
