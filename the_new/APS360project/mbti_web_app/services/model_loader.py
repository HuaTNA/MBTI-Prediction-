"""
Model Loading and Management Service
Centralized management for all ML models, question banks, and MBTI descriptions
"""

import os
import sys
import json
import pickle
import torch
import logging
from torchvision import transforms

# Add models directory to Python path
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

from models.emotion_model import EnhancedEmotionModel
from models.face_detector import FaceDetector, get_face_detector
from models.questions import QuestionBank
import config

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Centralized manager for all models and data
    Handles loading and initialization of:
    - Emotion recognition model
    - Text MBTI model
    - Face detector
    - Question banks
    - MBTI descriptions
    """

    def __init__(self):
        """Initialize model manager with empty state"""
        # Emotion recognition model
        self.emotion_model = None
        self.emotion_transform = None

        # Text MBTI model
        self.text_model = None
        self.text_vectorizer = None
        self.text_label_encoder = None

        # Face detector
        self.face_detector = None

        # Question banks
        self.question_bank_en = None
        self.question_bank_zh = None

        # MBTI descriptions
        self.mbti_descriptions_en = None
        self.mbti_descriptions_zh = None

        # Device for PyTorch models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Results directory
        os.makedirs(config.RESULTS_DIR, exist_ok=True)

    def load_emotion_model(self):
        """
        Load emotion recognition model

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create model instance
            self.emotion_model = EnhancedEmotionModel(
                num_classes=len(config.EMOTION_CATEGORIES),
                dropout_rates=config.EMOTION_MODEL_DROPOUT_RATES,
                backbone=config.EMOTION_MODEL_BACKBONE
            )

            # Load pretrained weights
            self.emotion_model.load_state_dict(
                torch.load(config.EMOTION_MODEL_PATH, map_location=self.device)
            )
            self.emotion_model.to(self.device)
            self.emotion_model.eval()
            logger.info(f"Successfully loaded emotion model: {config.EMOTION_MODEL_PATH}")

            # Define image transformation
            self.emotion_transform = transforms.Compose([
                transforms.Resize(config.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD)
            ])

            return True

        except Exception as e:
            logger.error(f"Error loading emotion model: {e}")
            return False

    def load_text_model(self):
        """
        Load text MBTI prediction model

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load model
            with open(config.TEXT_MODEL_PATH, 'rb') as f:
                self.text_model = pickle.load(f)

            # Load vectorizer
            with open(config.TEXT_VECTORIZER_PATH, 'rb') as f:
                self.text_vectorizer = pickle.load(f)

            # Load label encoder
            with open(config.TEXT_LABEL_ENCODER_PATH, 'rb') as f:
                self.text_label_encoder = pickle.load(f)

            # Load configuration
            with open(config.TEXT_CONFIG_PATH, 'r') as f:
                text_config = json.load(f)

            logger.info(f"Successfully loaded text model: {text_config.get('model_name', 'Unknown')}")
            return True

        except Exception as e:
            logger.error(f"Error loading text MBTI model: {e}")
            return False

    def load_face_detector(self):
        """
        Initialize face detector

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.face_detector = get_face_detector(
                prototxt_path=config.FACE_DETECTION_PROTOTXT,
                model_path=config.FACE_DETECTION_MODEL
            )
            logger.info("Successfully initialized OpenCV DNN face detector (replaces MediaPipe)")
            return True

        except Exception as e:
            logger.warning(f"Face detector initialization failed: {e}")
            logger.info("  Emotion recognition will be unavailable, but question management API will work")
            return False

    def load_question_banks(self):
        """
        Load question banks for both languages

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # English question bank
            if os.path.exists(config.QUESTIONS_EN_FILE):
                self.question_bank_en = QuestionBank(config.QUESTIONS_EN_FILE)
                logger.info(f"Successfully loaded English question bank: {len(self.question_bank_en.questions)} questions")
            else:
                logger.warning(f"English questions file not found: {config.QUESTIONS_EN_FILE}")
                self.question_bank_en = QuestionBank()

            # Chinese question bank
            if os.path.exists(config.QUESTIONS_ZH_FILE):
                self.question_bank_zh = QuestionBank(config.QUESTIONS_ZH_FILE)
                logger.info(f"Successfully loaded Chinese question bank: {len(self.question_bank_zh.questions)} questions")
            else:
                logger.warning(f"Chinese questions file not found: {config.QUESTIONS_ZH_FILE}")
                self.question_bank_zh = QuestionBank()

            return True

        except Exception as e:
            logger.error(f"Error loading question banks: {e}")
            # Create empty question banks to prevent crashes
            self.question_bank_en = QuestionBank()
            self.question_bank_zh = QuestionBank()
            return False

    def load_mbti_descriptions(self):
        """
        Load MBTI type descriptions for both languages

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # English descriptions
            if os.path.exists(config.MBTI_DESCRIPTIONS_EN_FILE):
                with open(config.MBTI_DESCRIPTIONS_EN_FILE, 'r', encoding='utf-8') as f:
                    self.mbti_descriptions_en = json.load(f)
                logger.info(f"Successfully loaded English MBTI descriptions: {len(self.mbti_descriptions_en.get('types', {}))} types")
            else:
                logger.warning(f"English MBTI descriptions file not found: {config.MBTI_DESCRIPTIONS_EN_FILE}")
                self.mbti_descriptions_en = {"types": {}}

            # Chinese descriptions
            if os.path.exists(config.MBTI_DESCRIPTIONS_ZH_FILE):
                with open(config.MBTI_DESCRIPTIONS_ZH_FILE, 'r', encoding='utf-8') as f:
                    self.mbti_descriptions_zh = json.load(f)
                logger.info(f"Successfully loaded Chinese MBTI descriptions: {len(self.mbti_descriptions_zh.get('types', {}))} types")
            else:
                logger.warning(f"Chinese MBTI descriptions file not found: {config.MBTI_DESCRIPTIONS_ZH_FILE}")
                self.mbti_descriptions_zh = {"types": {}}

            return True

        except Exception as e:
            logger.error(f"Error loading MBTI descriptions: {e}")
            self.mbti_descriptions_en = {"types": {}}
            self.mbti_descriptions_zh = {"types": {}}
            return False

    def initialize_all(self):
        """
        Initialize all models and data

        Returns:
            dict: Status of each component
        """
        logger.info("Initializing MBTI Assessment System...")

        status = {
            'face_detector': False,
            'emotion_model': False,
            'text_model': False,
            'question_banks': False,
            'mbti_descriptions': False
        }

        # Initialize face detector (optional)
        try:
            status['face_detector'] = self.load_face_detector()
        except Exception as e:
            logger.warning(f"Face detector initialization failed: {e}")

        # Load emotion model (optional)
        try:
            status['emotion_model'] = self.load_emotion_model()
        except Exception as e:
            logger.warning(f"Emotion model loading failed: {e}")

        # Load text model (optional)
        try:
            status['text_model'] = self.load_text_model()
        except Exception as e:
            logger.warning(f"Text model loading failed: {e}")

        # Load question banks (required)
        status['question_banks'] = self.load_question_banks()

        # Load MBTI descriptions
        status['mbti_descriptions'] = self.load_mbti_descriptions()

        # Print status summary
        logger.info("\n" + "="*60)
        logger.info("Component Loading Status:")
        logger.info(f"  {'✓' if status['face_detector'] else '✗'} Face Detector: {'Loaded' if status['face_detector'] else 'Not Loaded'}")
        logger.info(f"  {'✓' if status['emotion_model'] else '✗'} Emotion Model: {'Loaded' if status['emotion_model'] else 'Not Loaded'}")
        logger.info(f"  {'✓' if status['text_model'] else '✗'} Text MBTI Model: {'Loaded' if status['text_model'] else 'Not Loaded'}")
        logger.info(f"  {'✓' if status['question_banks'] else '✗'} Question Banks: {'Loaded' if status['question_banks'] else 'Not Loaded'}")
        logger.info(f"  {'✓' if status['mbti_descriptions'] else '✗'} MBTI Descriptions: {'Loaded' if status['mbti_descriptions'] else 'Not Loaded'}")
        logger.info("="*60)

        if status['question_banks']:
            logger.info("✓ Question Management API is ready!")
            logger.info("   Available endpoints: GET /api/questions, /api/questions/<id>, /api/questions/stats")

        if status['mbti_descriptions']:
            logger.info("✓ MBTI Description API is ready!")
            logger.info("   Available endpoints: GET /api/mbti/descriptions, /api/mbti/descriptions/<type>")

        if all([status['emotion_model'], status['text_model'], status['question_banks']]):
            logger.info("✓ All models and question banks loaded successfully! Full functionality available.")
        elif status['question_banks']:
            logger.info("⚠ Some features unavailable, but question management API is working")
            logger.info("   💡 Tip: You can still test the new question management features!")
        else:
            logger.error("✗ Question banks failed to load, system cannot start")
            sys.exit(1)

        return status


# Global model manager instance
_global_manager = None


def get_model_manager():
    """
    Get global model manager instance (singleton)

    Returns:
        ModelManager: Global model manager
    """
    global _global_manager
    if _global_manager is None:
        _global_manager = ModelManager()
    return _global_manager
