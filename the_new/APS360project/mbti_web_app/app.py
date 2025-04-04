# import os
# import cv2
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import pickle
# import json
# import base64
# import time
# import logging
# import mediapipe as mp
# from io import BytesIO
# from PIL import Image
# from flask import Flask, request, jsonify, render_template, send_from_directory
# from torchvision import transforms, models
# from datetime import datetime

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Set model file paths
# BASE_DIR = r"C:\Users\lnasl\Desktop\APS360\APS360\Model\TrainedModel"
# EMOTION_MODEL_PATH = os.path.join(BASE_DIR, "emotion", "improved_emotion_model.pth")
# TEXT_MODEL_DIR = os.path.join(BASE_DIR, "text", "ml")

# # Define emotion categories
# EMOTION_CATEGORIES = ['Anger', 'Confusion', 'Contempt', 'Disgust', 
#                       'Happiness', 'Neutral', 'Sadness', 'Surprise']

# # Create Flask application
# app = Flask(__name__)

# # Create directory for storing results
# RESULTS_DIR = os.path.join(os.path.expanduser("~"), "MBTI_Results")
# os.makedirs(RESULTS_DIR, exist_ok=True)

# # ================ MediaPipe Face Detection ================

# # Initialize MediaPipe face detection module
# mp_face_detection = mp.solutions.face_detection
# mp_drawing = mp.solutions.drawing_utils

# # Create global MediaPipe face detector
# face_detector = None

# def initialize_face_detector():
#     """Initialize MediaPipe face detector"""
#     global face_detector
#     face_detector = mp_face_detection.FaceDetection(
#         model_selection=1,  # 1 indicates full-range model, suitable for faces at a distance
#         min_detection_confidence=0.5  # Detection confidence threshold
#     )
#     logger.info("Successfully initialized MediaPipe face detector")

# def detect_and_crop_face(image, padding=0.2):
#     """
#     Detect and crop faces in an image
    
#     Parameters:
#         image: Input image (BGR)
#         padding: Additional padding around the bounding box, expressed as a proportion of the bounding box size
        
#     Returns:
#         On success, returns the cropped face image and True; on failure, returns the original image and False
#     """
#     global face_detector
    
#     # Ensure face detector is initialized
#     if face_detector is None:
#         initialize_face_detector()
    
#     # Convert BGR to RGB (MediaPipe requires RGB)
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     h, w, _ = image.shape
    
#     # Process image
#     results = face_detector.process(image_rgb)
    
#     # If face is detected
#     if results.detections:
#         # Use the detection with highest score (usually the largest/most central face)
#         detection = results.detections[0]
        
#         # Get bounding box
#         bbox = detection.location_data.relative_bounding_box
        
#         # Calculate absolute coordinates
#         xmin = max(0, int(bbox.xmin * w))
#         ymin = max(0, int(bbox.ymin * h))
#         width = min(int(bbox.width * w), w - xmin)
#         height = min(int(bbox.height * h), h - ymin)
        
#         # Calculate coordinates with padding
#         pad_x = int(width * padding)
#         pad_y = int(height * padding)
        
#         # Apply padding while ensuring it's within image boundaries
#         xmin = max(0, xmin - pad_x)
#         ymin = max(0, ymin - pad_y)
#         width = min(width + 2 * pad_x, w - xmin)
#         height = min(height + 2 * pad_y, h - ymin)
        
#         # Crop face
#         face_image = image[ymin:ymin+height, xmin:xmin+width]
        
#         return face_image, True
    
#     # If no face is detected, return original image
#     return image, False

# # ================ Model Definitions ================

# class SpatialAttention(nn.Module):
#     """Spatial Attention Module"""
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         assert kernel_size in (3, 7), 'Kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
        
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x_cat = torch.cat([avg_out, max_out], dim=1)
#         x_out = self.conv1(x_cat)
#         attention = self.sigmoid(x_out)
#         return x * attention

# class ChannelAttention(nn.Module):
#     """Channel Attention Module"""
#     def __init__(self, channels, reduction=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Conv2d(channels, channels // reduction, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels // reduction, channels, 1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()
        
#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         out = avg_out + max_out
#         attention = self.sigmoid(out)
#         return x * attention

# class CBAM(nn.Module):
#     """Convolutional Block Attention Module"""
#     def __init__(self, channels, reduction=16, kernel_size=7):
#         super(CBAM, self).__init__()
#         self.channel_attention = ChannelAttention(channels, reduction)
#         self.spatial_attention = SpatialAttention(kernel_size)
        
#     def forward(self, x):
#         x = self.channel_attention(x)
#         x = self.spatial_attention(x)
#         return x

# class EnhancedEmotionModel(nn.Module):
#     """Emotion Recognition CNN Model"""
#     def __init__(self, num_classes=8, dropout_rates=[0.5, 0.4, 0.3], backbone='efficientnet_b3'):
#         super(EnhancedEmotionModel, self).__init__()
        
#         # Select base model
#         if backbone == 'efficientnet_b3':
#             self.base_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
#             last_channel = self.base_model.classifier[1].in_features
#             self.base_model.classifier = nn.Identity()
#         else:
#             raise ValueError(f"Unsupported backbone network: {backbone}")
        
#         # Add attention module
#         self.cbam = CBAM(channels=last_channel, reduction=16, kernel_size=7)
        
#         # Global pooling after feature extraction
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
#         # Classification head
#         self.classifier = nn.Sequential(
#             nn.Dropout(dropout_rates[0]),
#             nn.Linear(last_channel, 1024),
#             nn.BatchNorm1d(1024),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(dropout_rates[1]),
#             nn.Linear(1024, 512),
#             nn.BatchNorm1d(512),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(dropout_rates[2]),
#             nn.Linear(512, num_classes)
#         )
        
#         # Specialized classifier for difficult categories
#         self.specialized_classifier = nn.Sequential(
#             nn.Dropout(0.3),
#             nn.Linear(last_channel, 512),
#             nn.BatchNorm1d(512),
#             nn.GELU(),
#             nn.Dropout(0.2),
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.GELU(),
#             nn.Linear(256, 3)
#         )
        
#     def forward(self, x):
#         # Feature extraction
#         features = self.base_model.features(x)
        
#         # Apply attention mechanism
#         features = self.cbam(features)
        
#         # Global pooling
#         features = self.avg_pool(features)
#         features = torch.flatten(features, 1)
        
#         # Main classifier output
#         main_output = self.classifier(features)
        
#         # In inference we only need the main output
#         return main_output, features

# # ================ Utility Functions ================

# def robust_text_preprocessing(text):
#     """Text preprocessing function"""
#     import re
    
#     # Ensure text is a string
#     text = str(text).lower()
    
#     # Remove URLs
#     text = re.sub(r'https?://\S+|www\.\S+', ' url ', text)
    
#     # Remove HTML tags
#     text = re.sub(r'<.*?>', ' ', text)
    
#     # Remove email addresses
#     text = re.sub(r'\S+@\S+', ' email ', text)
    
#     # Replace numbers with 'number' token
#     text = re.sub(r'\d+', ' number ', text)
    
#     # Process punctuation
#     text = re.sub(r'[^\w\s]', ' ', text)
    
#     # Simple stemming
#     words = text.split()
#     processed_words = []
    
#     # English stopwords list
#     stop_words = {
#         'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
#         'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
#         'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
#         'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
#         'further', 'then', 'once', 'here', 'there', 'all', 'any', 'both',
#         'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
#         'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
#         'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're',
#         've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn',
#         'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn',
#         'wasn', 'weren', 'won', 'wouldn', 'am', 'is', 'are', 'was', 'were',
#         'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
#         'did', 'doing'
#     }
    
#     for word in words:
#         # Skip stopwords
#         if word in stop_words:
#             continue
            
#         # Skip very short words
#         if len(word) <= 2:
#             continue
            
#         # Simple stemming rules
#         if word.endswith('ing') and len(word) > 4:
#             word = word[:-3]
#         elif word.endswith('ed') and len(word) > 3:
#             word = word[:-2]
#         elif word.endswith('ly') and len(word) > 3:
#             word = word[:-2]
#         elif word.endswith('ies') and len(word) > 4:
#             word = word[:-3] + 'y'
#         elif word.endswith('es') and len(word) > 3:
#             word = word[:-2]
#         elif word.endswith('s') and not word.endswith('ss') and len(word) > 2:
#             word = word[:-1]
            
#         processed_words.append(word)
    
#     # Rejoin words
#     text = ' '.join(processed_words)
    
#     # Remove extra spaces
#     text = re.sub(r'\s+', ' ', text).strip()
    
#     return text

# def process_base64_image(base64_image):
#     """Process base64 encoded image"""
#     # Remove data URL prefix (if exists)
#     if ',' in base64_image:
#         base64_image = base64_image.split(',')[1]
        
#     # Decode base64 image
#     image_data = base64.b64decode(base64_image)
    
#     # Convert to numpy array
#     nparr = np.frombuffer(image_data, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
#     return img

# # ================ Model Loading ================

# # Global variables to store loaded models
# emotion_model = None
# emotion_transform = None
# text_model = None
# text_vectorizer = None
# text_label_encoder = None
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def load_emotion_model():
#     """Load emotion recognition model"""
#     global emotion_model, emotion_transform
    
#     # Create model instance
#     emotion_model = EnhancedEmotionModel(num_classes=len(EMOTION_CATEGORIES))
    
#     try:
#         # Load pretrained weights
#         emotion_model.load_state_dict(torch.load(EMOTION_MODEL_PATH, map_location=device))
#         emotion_model.to(device)
#         emotion_model.eval()
#         logger.info(f"Successfully loaded emotion model: {EMOTION_MODEL_PATH}")
        
#         # Define image transformations
#         emotion_transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
        
#         return True
        
#     except Exception as e:
#         logger.error(f"Error loading emotion model: {e}")
#         return False

# def load_text_model():
#     """Load text MBTI model"""
#     global text_model, text_vectorizer, text_label_encoder
    
#     try:
#         # Load model
#         with open(os.path.join(TEXT_MODEL_DIR, 'model.pkl'), 'rb') as f:
#             text_model = pickle.load(f)
        
#         # Load vectorizer
#         with open(os.path.join(TEXT_MODEL_DIR, 'vectorizer.pkl'), 'rb') as f:
#             text_vectorizer = pickle.load(f)
        
#         # Load label encoder
#         with open(os.path.join(TEXT_MODEL_DIR, 'label_encoder.pkl'), 'rb') as f:
#             text_label_encoder = pickle.load(f)
        
#         # Load configuration
#         with open(os.path.join(TEXT_MODEL_DIR, 'config.json'), 'r') as f:
#             text_config = json.load(f)
            
#         logger.info(f"Successfully loaded text model: {text_config.get('model_name', 'Unknown')}")
#         return True
        
#     except Exception as e:
#         logger.error(f"Error loading text MBTI model: {e}")
#         return False

# # ================ Emotion Analysis Features ================

# def analyze_emotion(image):
#     """Analyze image using emotion CNN model"""
#     global emotion_model, emotion_transform, device
    
#     try:
#         # Detect and crop face
#         face_image, face_detected = detect_and_crop_face(image)
        
#         if not face_detected:
#             logger.warning("No face detected, using the entire image for analysis")
        
#         # Convert to RGB (OpenCV uses BGR)
#         image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
#         # Convert to PIL image
#         pil_img = Image.fromarray(image_rgb)
        
#         # Apply transformations
#         input_tensor = emotion_transform(pil_img)
#         input_tensor = input_tensor.unsqueeze(0).to(device)
        
#         # Inference
#         with torch.no_grad():
#             predictions, _ = emotion_model(input_tensor)
#             probabilities = F.softmax(predictions, dim=1)[0].cpu().numpy()
        
#         # Create emotion dictionary
#         emotion_dict = {emotion: float(prob) for emotion, prob in zip(EMOTION_CATEGORIES, probabilities)}
        
#         return emotion_dict, face_detected
        
#     except Exception as e:
#         logger.error(f"Error in emotion analysis: {e}")
#         # Return average probabilities (fallback)
#         return {emotion: 1.0/len(EMOTION_CATEGORIES) for emotion in EMOTION_CATEGORIES}, False

# # ================ Text MBTI Analysis Features ================

# def analyze_mbti_text(text):
#     """Analyze text to predict MBTI"""
#     global text_model, text_vectorizer, text_label_encoder
    
#     # Preprocess text
#     processed_text = robust_text_preprocessing(text)
    
#     # Convert to feature vector
#     text_vector = text_vectorizer.transform([processed_text])
    
#     # Predict
#     if hasattr(text_model, 'predict_proba'):
#         # Get prediction probabilities
#         proba = text_model.predict_proba(text_vector)[0]
#         prediction = proba.argmax()
#         mbti_type = text_label_encoder.inverse_transform([prediction])[0]
        
#         # Extract dimension scores
#         dimension_scores = extract_dimension_scores(proba)
        
#         return {
#             'mbti_type': mbti_type,
#             'dimension_scores': dimension_scores,
#             'confidence': float(proba[prediction])
#         }
#     else:
#         # Model without probability support
#         prediction = text_model.predict(text_vector)[0]
#         mbti_type = text_label_encoder.inverse_transform([prediction])[0]
        
#         # Create placeholder dimension scores
#         dimension_scores = extract_dimension_scores_from_type(mbti_type)
        
#         return {
#             'mbti_type': mbti_type,
#             'dimension_scores': dimension_scores,
#             'confidence': 1.0
#         }

# def extract_dimension_scores(probabilities):
#     """Extract dimension scores from model probabilities (I/E, S/N, T/F, J/P)"""
#     # Initialize dimension scores
#     ie_scores = [0.0, 0.0]  # [I, E]
#     sn_scores = [0.0, 0.0]  # [S, N]
#     tf_scores = [0.0, 0.0]  # [T, F]
#     jp_scores = [0.0, 0.0]  # [J, P]
    
#     # Iterate through all types and their probabilities
#     for i, type_prob in enumerate(probabilities):
#         mbti_type = text_label_encoder.inverse_transform([i])[0]
        
#         # Add probability to respective dimension scores
#         ie_scores[0 if mbti_type[0] == 'I' else 1] += type_prob
#         sn_scores[0 if mbti_type[1] == 'S' else 1] += type_prob
#         tf_scores[0 if mbti_type[2] == 'T' else 1] += type_prob
#         jp_scores[0 if mbti_type[3] == 'J' else 1] += type_prob
    
#     # Create dimension scores dictionary
#     dimension_scores = {
#         "I/E": (ie_scores[0], ie_scores[1]),  # (I, E)
#         "S/N": (sn_scores[0], sn_scores[1]),  # (S, N)
#         "T/F": (tf_scores[0], tf_scores[1]),  # (T, F)
#         "J/P": (jp_scores[0], jp_scores[1])   # (J, P)
#     }
    
#     return dimension_scores

# def extract_dimension_scores_from_type(mbti_type):
#     """Create dimension scores from a single MBTI type (for models without probability support)"""
#     dimension_scores = {
#         "I/E": (0.8, 0.2) if mbti_type[0] == 'I' else (0.2, 0.8),
#         "S/N": (0.8, 0.2) if mbti_type[1] == 'S' else (0.2, 0.8),
#         "T/F": (0.8, 0.2) if mbti_type[2] == 'T' else (0.2, 0.8),
#         "J/P": (0.8, 0.2) if mbti_type[3] == 'J' else (0.2, 0.8)
#     }
#     return dimension_scores

# # ================ Multimodal Integration ================

# # Modified integrate_multimodal_data function to support more questions

# def integrate_multimodal_data(responses):
#     """Integrate text and emotion data from all questions to predict final MBTI"""
    
#     # Dimension mapping record - no longer limited to 3 questions
#     dimension_focus_count = {
#         "I/E": 0,
#         "S/N": 0,
#         "T/F": 0,
#         "J/P": 0
#     }
    
#     # Initialize final dimension scores
#     final_dimension_scores = {
#         "I/E": [0, 0],
#         "S/N": [0, 0],
#         "T/F": [0, 0],
#         "J/P": [0, 0]
#     }
    
#     # Process each question's results
#     for response in responses:
#         q_idx = response.get('questionIndex', 0)
        
#         # Get question's dimension focus
#         question_dimension = response.get('questionDimension', None)
#         if question_dimension and question_dimension in dimension_focus_count:
#             dimension_focus_count[question_dimension] += 1
        
#         # Get text MBTI analysis results
#         text_mbti = response.get('mbtiResults', {})
#         dimension_scores = text_mbti.get('dimension_scores', {})
        
#         # Get emotion distribution
#         emotion_data = response.get('emotionData', [])
        
#         # Calculate average emotion distribution
#         emotion_sum = {}
#         for data_point in emotion_data:
#             emotions = data_point.get('emotions', {})
#             for emotion, value in emotions.items():
#                 emotion_sum[emotion] = emotion_sum.get(emotion, 0) + value
        
#         # Normalize emotion distribution
#         total = sum(emotion_sum.values()) if emotion_sum else 1
#         emotion_distribution = {e: v / total for e, v in emotion_sum.items()}
        
#         # Apply text-based scores for the current question's dimensions
#         focus = question_dimension  # Use question's dimension focus
        
#         for dimension, scores in dimension_scores.items():
#             # Give higher weight to question's focus dimension
#             weight = 2.0 if dimension == focus else 1.0
#             if isinstance(scores, (list, tuple)) and len(scores) == 2:
#                 final_dimension_scores[dimension][0] += scores[0] * weight
#                 final_dimension_scores[dimension][1] += scores[1] * weight
        
#         # Apply emotion adjustments
#         # Extract emotion percentages
#         happiness = emotion_distribution.get('Happiness', 0)
#         surprise = emotion_distribution.get('Surprise', 0)
#         confusion = emotion_distribution.get('Confusion', 0)
#         neutral = emotion_distribution.get('Neutral', 0)
#         sadness = emotion_distribution.get('Sadness', 0)
#         anger = emotion_distribution.get('Anger', 0)
#         contempt = emotion_distribution.get('Contempt', 0)
#         disgust = emotion_distribution.get('Disgust', 0)
        
#         # ===== I/E dimension - Introverts tend to show more controlled emotions overall =====
#         # Extraverts tend to express emotions more openly and with higher intensity
#         emotional_expressiveness = happiness + surprise + sadness + anger - neutral
#         if emotional_expressiveness > 0.3:
#             # Higher expressiveness suggests Extraversion
#             adjust_factor = 0.09 if focus == "I/E" else 0.04
#             e_boost = min(adjust_factor, emotional_expressiveness / 3)
#             final_dimension_scores["I/E"][0] -= e_boost  # Decrease I
#             final_dimension_scores["I/E"][1] += e_boost  # Increase E
#         elif neutral > 0.5:
#             # Higher neutral expression suggests Introversion
#             adjust_factor = 0.09 if focus == "I/E" else 0.04
#             i_boost = min(adjust_factor, neutral / 3)
#             final_dimension_scores["I/E"][0] += i_boost  # Increase I
#             final_dimension_scores["I/E"][1] -= i_boost  # Decrease E
        
#         # ===== S/N dimension - Sensing vs Intuition patterns =====
#         # Intuitive types often show more curiosity, wonder, and comfort with ambiguity
#         if surprise > 0.3 and confusion > 0.2:
#             # Surprise + confusion suggests openness to new ideas/abstraction (Intuition)
#             adjust_factor = 0.11 if focus == "S/N" else 0.04
#             n_boost = min(adjust_factor, (surprise + confusion) / 4)
#             final_dimension_scores["S/N"][0] -= n_boost  # Decrease S
#             final_dimension_scores["S/N"][1] += n_boost  # Increase N
#         elif neutral > 0.4 and happiness > 0.3:
#             # Neutral + happiness suggests present-focused attention (Sensing)
#             adjust_factor = 0.09 if focus == "S/N" else 0.04
#             s_boost = min(adjust_factor, (neutral + happiness) / 5)
#             final_dimension_scores["S/N"][0] += s_boost  # Increase S
#             final_dimension_scores["S/N"][1] -= s_boost  # Decrease N
        
#         # ===== T/F dimension - Thinking vs Feeling patterns =====
#         # Feeling types typically display more varied emotional expression
#         # Thinking types often display more controlled emotions or analytical expressions
#         emotional_depth = sadness + happiness - (contempt + disgust)
#         if emotional_depth > 0.2:
#             # Higher emotional depth suggests Feeling preference
#             adjust_factor = 0.2 if focus == "T/F" else 0.08
#             f_boost = min(adjust_factor, emotional_depth / 3)
#             final_dimension_scores["T/F"][0] -= f_boost  # Decrease T
#             final_dimension_scores["T/F"][1] += f_boost  # Increase F
#         elif contempt + disgust > 0.2 or (neutral > 0.6 and anger > 0.2):
#             # Contempt/disgust or controlled anger suggests Thinking preference
#             adjust_factor = 0.18 if focus == "T/F" else 0.07
#             t_boost = min(adjust_factor, (contempt + disgust + anger) / 4)
#             final_dimension_scores["T/F"][0] += t_boost  # Increase T
#             final_dimension_scores["T/F"][1] -= t_boost  # Decrease F
        
#         # ===== J/P dimension - Judging vs Perceiving patterns =====
#         # Perceiving types are more comfortable with ambiguity and spontaneity
#         # Judging types prefer structure, closure, and can be more decisive
#         if confusion > 0.25 and surprise > 0.25:
#             # High confusion + surprise suggests comfort with ambiguity (Perceiving)
#             adjust_factor = 0.18 if focus == "J/P" else 0.07
#             p_boost = min(adjust_factor, (confusion + surprise) / 4)
#             final_dimension_scores["J/P"][0] -= p_boost  # Decrease J
#             final_dimension_scores["J/P"][1] += p_boost  # Increase P
#         elif neutral > 0.4 and contempt > 0.15:
#             # Neutral + contempt suggests preference for order (Judging)
#             adjust_factor = 0.15 if focus == "J/P" else 0.06
#             j_boost = min(adjust_factor, (neutral + contempt) / 4)
#             final_dimension_scores["J/P"][0] += j_boost  # Increase J
#             final_dimension_scores["J/P"][1] -= j_boost  # Decrease P
    
#     # Normalize scores
#     for dimension in final_dimension_scores:
#         scores = final_dimension_scores[dimension]
#         total = sum(scores)
#         if total > 0:
#             final_dimension_scores[dimension] = [s/total for s in scores]
    
#     # Determine final MBTI type
#     mbti_type = ""
#     mbti_type += "I" if final_dimension_scores["I/E"][0] > final_dimension_scores["I/E"][1] else "E"
#     mbti_type += "S" if final_dimension_scores["S/N"][0] > final_dimension_scores["S/N"][1] else "N"
#     mbti_type += "T" if final_dimension_scores["T/F"][0] > final_dimension_scores["T/F"][1] else "F"
#     mbti_type += "J" if final_dimension_scores["J/P"][0] > final_dimension_scores["J/P"][1] else "P"
    
#     # Prepare MBTI type description data
#     mbti_descriptions = {
#         "ISTJ": "The Inspector - A practical, disciplined, and reliable personality type.",
#         "ISFJ": "The Protector - A caring, loyal, and responsible personality type.",
#         "INFJ": "The Counselor - An insightful, idealistic, and creative personality type.",
#         "INTJ": "The Mastermind - An independent, strategic, and innovative personality type.",
#         "ISTP": "The Craftsman - A practical, adaptable, and independent personality type.",
#         "ISFP": "The Composer - A gentle, sensitive, and artistic personality type.",
#         "INFP": "The Healer - An idealistic, compassionate, and creative personality type.",
#         "INTP": "The Architect - A logical, analytical, and innovative personality type.",
#         "ESTP": "The Dynamo - An energetic, practical, and adaptable personality type.",
#         "ESFP": "The Performer - An outgoing, friendly, and life-loving personality type.",
#         "ENFP": "The Champion - An enthusiastic, creative, and sociable personality type.",
#         "ENTP": "The Debater - An innovative, intelligent, and energetic personality type.",
#         "ESTJ": "The Supervisor - An organized, responsible, and practical personality type.",
#         "ESFJ": "The Provider - A friendly, cooperative, and harmony-valuing personality type.",
#         "ENFJ": "The Teacher - A compassionate, influential, and organized personality type.",
#         "ENTJ": "The Commander - A decisive, strategic, and leadership-capable personality type."
#     }
    
#     return {
#         'mbti_type': mbti_type,
#         'mbti_description': mbti_descriptions.get(mbti_type, "Unknown Type"),
#         'dimension_scores': final_dimension_scores,
#         'dimension_focus_count': dimension_focus_count
#     }
# # ================ Save Results ================

# def save_assessment_results(results):
#     """Save assessment results to file"""
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     file_path = os.path.join(RESULTS_DIR, f"mbti_assessment_{timestamp}.json")
    
#     with open(file_path, "w", encoding="utf-8") as f:
#         json.dump(results, f, indent=2, ensure_ascii=False)
    
#     return file_path

# # ================ Flask Routes ================

# @app.route('/')
# def index():
#     """Render homepage"""
#     return render_template('index.html')

# @app.route('/api/process_frame', methods=['POST'])
# def process_frame():
#     """Process video frame for emotion analysis"""
#     try:
#         # Receive image from frontend
#         data = request.json
#         image_data = data.get('image', '')
        
#         # Process base64 image
#         if image_data:
#             img = process_base64_image(image_data)
            
#             # Analyze emotion
#             emotions, face_detected = analyze_emotion(img)
            
#             return jsonify({
#                 'emotions': emotions, 
#                 'face_detected': face_detected
#             })
#         else:
#             return jsonify({'error': 'No image data received'}), 400
            
#     except Exception as e:
#         logger.error(f"Error processing frame: {e}")
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/process_text', methods=['POST'])
# def process_text():
#     """Process text for MBTI analysis"""
#     try:
#         # Receive text from frontend
#         data = request.json
#         text = data.get('text', '')
        
#         if text:
#             # Analyze MBTI
#             mbti_results = analyze_mbti_text(text)
            
#             return jsonify(mbti_results)
#         else:
#             return jsonify({'error': 'No text data received'}), 400
            
#     except Exception as e:
#         logger.error(f"Error processing text: {e}")
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/process_speech', methods=['POST'])
# def process_speech():
#     """Process speech recognition results"""
#     try:
#         # Receive speech recognition text from frontend
#         data = request.json
#         speech_text = data.get('speech_text', '')
        
#         if speech_text:
#             # Log original speech recognition text
#             logger.info(f"Received speech recognition text: {speech_text}")
            
#             # Use same method as text analysis to analyze MBTI
#             mbti_results = analyze_mbti_text(speech_text)
            
#             return jsonify({
#                 'original_text': speech_text,
#                 'mbti_results': mbti_results
#             })
#         else:
#             return jsonify({'error': 'No speech recognition text received'}), 400
            
#     except Exception as e:
#         logger.error(f"Error processing speech recognition results: {e}")
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/final_results', methods=['POST'])
# def generate_final_results():
#     """Generate final MBTI prediction results"""
#     try:
#         # Receive all question responses and emotion data
#         data = request.json
#         responses = data.get('responses', [])
        
#         if responses:
#             # Comprehensive analysis to generate final prediction
#             final_results = integrate_multimodal_data(responses)
            
#             # Save results
#             file_path = save_assessment_results({
#                 "timestamp": datetime.now().isoformat(),
#                 "responses": responses,
#                 "final_results": final_results
#             })
            
#             # Add result save path
#             final_results['saved_to'] = file_path
            
#             return jsonify(final_results)
#         else:
#             return jsonify({'error': 'No response data received'}), 400
            
#     except Exception as e:
#         logger.error(f"Error generating final results: {e}")
#         return jsonify({'error': str(e)}), 500

# @app.route('/results/<path:filename>')
# def download_results(filename):
#     """Allow downloading saved result files"""
#     return send_from_directory(RESULTS_DIR, filename, as_attachment=True)

# @app.route('/test')
# def test_page():
#     """Test page to verify server is running properly"""
#     return "MBTI Assessment System server is running normally!"

# # ================ Application Startup ================

# def initialize_app():
#     """Initialize application, load all models"""
#     logger.info("Initializing MBTI Assessment System...")
    
#     # Initialize face detector
#     initialize_face_detector()
    
#     # Load emotion model
#     emotion_model_loaded = load_emotion_model()
    
#     # Load text model
#     text_model_loaded = load_text_model()
    
#     if emotion_model_loaded and text_model_loaded:
#         logger.info("All models loaded successfully! System is ready.")
#     else:
#         logger.warning("Warning: Some models failed to load. System may not function properly.")

# if __name__ == '__main__':
#     # Initialize application
#     initialize_app()
    
#     # Start server
#     app.run(debug=True, host='0.0.0.0', port=5000)





import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import json
import base64
import time
import logging
import mediapipe as mp
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory
from torchvision import transforms, models
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置模型文件路径
BASE_DIR = r"C:\Users\lnasl\Desktop\APS360project\mbti_web_app\models"
EMOTION_MODEL_PATH = os.path.join(BASE_DIR, "emotion", "improved_emotion_model.pth")
TEXT_MODEL_DIR = os.path.join(BASE_DIR, "text", "ml")

# 定义情绪类别
EMOTION_CATEGORIES = ['Anger', 'Confusion', 'Contempt', 'Disgust', 
                      'Happiness', 'Neutral', 'Sadness', 'Surprise']

# 创建 Flask 应用
app = Flask(__name__)

# 为存储结果创建目录
RESULTS_DIR = os.path.join(os.path.expanduser("~"), "MBTI_Results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ================ MediaPipe 人脸检测 ================

# 初始化 MediaPipe 人脸检测模块
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 创建全局 MediaPipe 人脸检测器
face_detector = None

def initialize_face_detector():
    """初始化 MediaPipe 人脸检测器"""
    global face_detector
    face_detector = mp_face_detection.FaceDetection(
        model_selection=1,  # 1 表示完整范围模型，适合距离远的人脸
        min_detection_confidence=0.5  # 检测置信度阈值
    )
    logger.info("成功初始化 MediaPipe 人脸检测器")

def detect_and_crop_face(image, padding=0.2):
    """
    检测并裁剪图像中的人脸
    
    参数:
        image: 输入图像 (BGR)
        padding: 边界框周围的额外填充，表示为边界框大小的比例
        
    返回:
        成功时返回裁剪后的人脸图像和True，失败时返回原始图像和False
    """
    global face_detector
    
    # 确保人脸检测器已初始化
    if face_detector is None:
        initialize_face_detector()
    
    # 转换 BGR 到 RGB (MediaPipe 需要 RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    
    # 处理图像
    results = face_detector.process(image_rgb)
    
    # 如果检测到人脸
    if results.detections:
        # 使用得分最高的人脸（通常是最大/最中心的人脸）
        detection = results.detections[0]
        
        # 获取边界框
        bbox = detection.location_data.relative_bounding_box
        
        # 计算绝对坐标
        xmin = max(0, int(bbox.xmin * w))
        ymin = max(0, int(bbox.ymin * h))
        width = min(int(bbox.width * w), w - xmin)
        height = min(int(bbox.height * h), h - ymin)
        
        # 计算带填充的坐标
        pad_x = int(width * padding)
        pad_y = int(height * padding)
        
        # 应用填充但确保在图像边界内
        xmin = max(0, xmin - pad_x)
        ymin = max(0, ymin - pad_y)
        width = min(width + 2 * pad_x, w - xmin)
        height = min(height + 2 * pad_y, h - ymin)
        
        # 裁剪人脸
        face_image = image[ymin:ymin+height, xmin:xmin+width]
        
        return face_image, True
    
    # 如果未检测到人脸，返回原始图像
    return image, False

# ================ 模型定义 ================

class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), '内核大小必须为 3 或 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv1(x_cat)
        attention = self.sigmoid(x_out)
        return x * attention

class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        attention = self.sigmoid(out)
        return x * attention

class CBAM(nn.Module):
    """卷积块注意力模块"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class EnhancedEmotionModel(nn.Module):
    """情绪识别 CNN 模型"""
    def __init__(self, num_classes=8, dropout_rates=[0.5, 0.4, 0.3], backbone='efficientnet_b3'):
        super(EnhancedEmotionModel, self).__init__()
        
        # 选择基础模型
        if backbone == 'efficientnet_b3':
            self.base_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
            last_channel = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Identity()
        else:
            raise ValueError(f"不支持的骨干网络: {backbone}")
        
        # 添加注意力模块
        self.cbam = CBAM(channels=last_channel, reduction=16, kernel_size=7)
        
        # 特征提取后的全局池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rates[0]),
            nn.Linear(last_channel, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rates[1]),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rates[2]),
            nn.Linear(512, num_classes)
        )
        
        # 难分类别的专门分类头
        self.specialized_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(last_channel, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 3)
        )
        
    def forward(self, x):
        # 特征提取
        features = self.base_model.features(x)
        
        # 应用注意力机制
        features = self.cbam(features)
        
        # 全局池化
        features = self.avg_pool(features)
        features = torch.flatten(features, 1)
        
        # 主分类器输出
        main_output = self.classifier(features)
        
        # 在推理中我们只需要主输出
        return main_output, features

# ================ 工具函数 ================

def robust_text_preprocessing(text):
    """文本预处理函数"""
    import re
    
    # 确保文本是字符串
    text = str(text).lower()
    
    # 移除 URL
    text = re.sub(r'https?://\S+|www\.\S+', ' url ', text)
    
    # 移除 HTML 标签
    text = re.sub(r'<.*?>', ' ', text)
    
    # 移除电子邮件地址
    text = re.sub(r'\S+@\S+', ' email ', text)
    
    # 替换数字为 'number' 标记
    text = re.sub(r'\d+', ' number ', text)
    
    # 处理标点符号
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # 简单词干提取
    words = text.split()
    processed_words = []
    
    # 英语停用词列表
    stop_words = {
        'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
        'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
        'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
        'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'all', 'any', 'both',
        'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
        'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
        'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're',
        've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn',
        'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn',
        'wasn', 'weren', 'won', 'wouldn', 'am', 'is', 'are', 'was', 'were',
        'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
        'did', 'doing'
    }
    
    for word in words:
        # 跳过停用词
        if word in stop_words:
            continue
            
        # 跳过极短词
        if len(word) <= 2:
            continue
            
        # 简单词干提取规则
        if word.endswith('ing') and len(word) > 4:
            word = word[:-3]
        elif word.endswith('ed') and len(word) > 3:
            word = word[:-2]
        elif word.endswith('ly') and len(word) > 3:
            word = word[:-2]
        elif word.endswith('ies') and len(word) > 4:
            word = word[:-3] + 'y'
        elif word.endswith('es') and len(word) > 3:
            word = word[:-2]
        elif word.endswith('s') and not word.endswith('ss') and len(word) > 2:
            word = word[:-1]
            
        processed_words.append(word)
    
    # 重新连接词语
    text = ' '.join(processed_words)
    
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def process_base64_image(base64_image):
    """处理 base64 编码的图像"""
    # 移除 data URL 前缀 (如果存在)
    if ',' in base64_image:
        base64_image = base64_image.split(',')[1]
        
    # 解码 base64 图像
    image_data = base64.b64decode(base64_image)
    
    # 转换为 numpy 数组
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    return img

# ================ 模型加载 ================

# 全局变量存储加载的模型
emotion_model = None
emotion_transform = None
text_model = None
text_vectorizer = None
text_label_encoder = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_emotion_model():
    """加载情绪识别模型"""
    global emotion_model, emotion_transform
    
    # 创建模型实例
    emotion_model = EnhancedEmotionModel(num_classes=len(EMOTION_CATEGORIES))
    
    try:
        # 加载预训练权重
        emotion_model.load_state_dict(torch.load(EMOTION_MODEL_PATH, map_location=device))
        emotion_model.to(device)
        emotion_model.eval()
        logger.info(f"成功加载情绪模型: {EMOTION_MODEL_PATH}")
        
        # 定义图像变换
        emotion_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return True
        
    except Exception as e:
        logger.error(f"加载情绪模型时出错: {e}")
        return False

def load_text_model():
    """加载文本 MBTI 模型"""
    global text_model, text_vectorizer, text_label_encoder
    
    try:
        # 加载模型
        with open(os.path.join(TEXT_MODEL_DIR, 'model.pkl'), 'rb') as f:
            text_model = pickle.load(f)
        
        # 加载向量化器
        with open(os.path.join(TEXT_MODEL_DIR, 'vectorizer.pkl'), 'rb') as f:
            text_vectorizer = pickle.load(f)
        
        # 加载标签编码器
        with open(os.path.join(TEXT_MODEL_DIR, 'label_encoder.pkl'), 'rb') as f:
            text_label_encoder = pickle.load(f)
        
        # 加载配置
        with open(os.path.join(TEXT_MODEL_DIR, 'config.json'), 'r') as f:
            text_config = json.load(f)
            
        logger.info(f"成功加载文本模型: {text_config.get('model_name', '未知')}")
        return True
        
    except Exception as e:
        logger.error(f"加载文本 MBTI 模型时出错: {e}")
        return False

# ================ 情绪分析功能 ================

def analyze_emotion(image):
    """使用情绪 CNN 模型分析图像"""
    global emotion_model, emotion_transform, device
    
    try:
        # 检测并裁剪人脸
        face_image, face_detected = detect_and_crop_face(image)
        
        if not face_detected:
            logger.warning("未检测到人脸，使用整个图像进行分析")
        
        # 转换为 RGB (OpenCV 使用 BGR)
        image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # 转换为 PIL 图像
        pil_img = Image.fromarray(image_rgb)
        
        # 应用变换
        input_tensor = emotion_transform(pil_img)
        input_tensor = input_tensor.unsqueeze(0).to(device)
        
        # 推理
        with torch.no_grad():
            predictions, _ = emotion_model(input_tensor)
            probabilities = F.softmax(predictions, dim=1)[0].cpu().numpy()
        
        # 创建情绪字典
        emotion_dict = {emotion: float(prob) for emotion, prob in zip(EMOTION_CATEGORIES, probabilities)}
        
        return emotion_dict, face_detected
        
    except Exception as e:
        logger.error(f"情绪分析出错: {e}")
        # 返回平均概率（回退）
        return {emotion: 1.0/len(EMOTION_CATEGORIES) for emotion in EMOTION_CATEGORIES}, False

# ================ 文本 MBTI 分析功能 ================

def analyze_mbti_text(text):
    """分析文本预测 MBTI"""
    global text_model, text_vectorizer, text_label_encoder
    
    # 预处理文本
    processed_text = robust_text_preprocessing(text)
    
    # 转换为特征向量
    text_vector = text_vectorizer.transform([processed_text])
    
    # 预测
    if hasattr(text_model, 'predict_proba'):
        # 获取预测概率
        proba = text_model.predict_proba(text_vector)[0]
        prediction = proba.argmax()
        mbti_type = text_label_encoder.inverse_transform([prediction])[0]
        
        # 提取各维度评分
        dimension_scores = extract_dimension_scores(proba)
        
        return {
            'mbti_type': mbti_type,
            'dimension_scores': dimension_scores,
            'confidence': float(proba[prediction])
        }
    else:
        # 不支持概率的模型
        prediction = text_model.predict(text_vector)[0]
        mbti_type = text_label_encoder.inverse_transform([prediction])[0]
        
        # 创建占位符维度评分
        dimension_scores = extract_dimension_scores_from_type(mbti_type)
        
        return {
            'mbti_type': mbti_type,
            'dimension_scores': dimension_scores,
            'confidence': 1.0
        }

def extract_dimension_scores(probabilities):
    """从模型概率中提取维度评分 (I/E, S/N, T/F, J/P)"""
    # 初始化各维度评分
    ie_scores = [0.0, 0.0]  # [I, E]
    sn_scores = [0.0, 0.0]  # [S, N]
    tf_scores = [0.0, 0.0]  # [T, F]
    jp_scores = [0.0, 0.0]  # [J, P]
    
    # 遍历所有类型及其概率
    for i, type_prob in enumerate(probabilities):
        mbti_type = text_label_encoder.inverse_transform([i])[0]
        
        # 将概率添加到相应的维度评分
        ie_scores[0 if mbti_type[0] == 'I' else 1] += type_prob
        sn_scores[0 if mbti_type[1] == 'S' else 1] += type_prob
        tf_scores[0 if mbti_type[2] == 'T' else 1] += type_prob
        jp_scores[0 if mbti_type[3] == 'J' else 1] += type_prob
    
    # 创建维度评分字典
    dimension_scores = {
        "I/E": (ie_scores[0], ie_scores[1]),  # (I, E)
        "S/N": (sn_scores[0], sn_scores[1]),  # (S, N)
        "T/F": (tf_scores[0], tf_scores[1]),  # (T, F)
        "J/P": (jp_scores[0], jp_scores[1])   # (J, P)
    }
    
    return dimension_scores

def extract_dimension_scores_from_type(mbti_type):
    """从单一 MBTI 类型创建维度评分（用于不支持概率的模型）"""
    dimension_scores = {
        "I/E": (0.8, 0.2) if mbti_type[0] == 'I' else (0.2, 0.8),
        "S/N": (0.8, 0.2) if mbti_type[1] == 'S' else (0.2, 0.8),
        "T/F": (0.8, 0.2) if mbti_type[2] == 'T' else (0.2, 0.8),
        "J/P": (0.8, 0.2) if mbti_type[3] == 'J' else (0.2, 0.8)
    }
    return dimension_scores

# ================ 多模态整合 ================

def integrate_multimodal_data(responses):
    """整合所有问题的文本和情绪数据来预测最终 MBTI"""
    
    # 问题焦点维度映射
    question_focus = [
        "I/E",  # 问题 1 关注内向/外向
        "S/N",  # 问题 2 关注感觉/直觉
        "J/P"   # 问题 3 关注判断/感知
    ]
    
    # 初始化最终维度评分
    final_dimension_scores = {
        "I/E": [0, 0],
        "S/N": [0, 0],
        "T/F": [0, 0],
        "J/P": [0, 0]
    }
    
    # 处理每个问题的结果
    for response in responses:
        q_idx = response.get('questionIndex', 0)
        
        # 获取文本 MBTI 分析结果
        text_mbti = response.get('mbtiResults', {})
        dimension_scores = text_mbti.get('dimension_scores', {})
        
        # 获取情绪分布
        emotion_data = response.get('emotionData', [])
        
        # 计算平均情绪分布
        emotion_sum = {}
        for data_point in emotion_data:
            emotions = data_point.get('emotions', {})
            for emotion, value in emotions.items():
                emotion_sum[emotion] = emotion_sum.get(emotion, 0) + value
        
        # 归一化情绪分布
        total = sum(emotion_sum.values()) if emotion_sum else 1
        emotion_distribution = {e: v / total for e, v in emotion_sum.items()}
        
        # 对当前问题的各维度应用基于文本的评分
        focus = question_focus[q_idx] if q_idx < len(question_focus) else None
        
        for dimension, scores in dimension_scores.items():
            # 给与问题焦点维度更高的权重
            weight = 2.0 if dimension == focus else 1.0
            if isinstance(scores, (list, tuple)) and len(scores) == 2:
                final_dimension_scores[dimension][0] += scores[0] * weight
                final_dimension_scores[dimension][1] += scores[1] * weight
        
        # 应用情绪调整
        # 提取情绪百分比
        happiness = emotion_distribution.get('Happiness', 0)
        surprise = emotion_distribution.get('Surprise', 0)
        confusion = emotion_distribution.get('Confusion', 0)
        neutral = emotion_distribution.get('Neutral', 0)
        sadness = emotion_distribution.get('Sadness', 0)
        anger = emotion_distribution.get('Anger', 0)
        
        # I/E 维度 - 基于 Happiness 和 Surprise 调整
        if happiness + surprise > 0.4 and neutral < 0.4:
            # 如果问题关注 I/E 则调整更大
            adjust_factor = 0.15 if focus == "I/E" else 0.05
            e_boost = min(adjust_factor, (happiness + surprise - 0.4) / 2)
            final_dimension_scores["I/E"][0] -= e_boost  # 减少 I
            final_dimension_scores["I/E"][1] += e_boost  # 增加 E
        
        # S/N 维度 - 基于 Confusion 调整
        if confusion > 0.2:
            adjust_factor = 0.15 if focus == "S/N" else 0.05
            n_penalty = min(adjust_factor, confusion / 10)
            final_dimension_scores["S/N"][0] += n_penalty  # 增加 S
            final_dimension_scores["S/N"][1] -= n_penalty  # 减少 N
        
        # T/F 维度 - 基于情绪表现调整
        emotional_sum = sadness + anger + happiness
        if emotional_sum > 0.3:
            adjust_factor = 0.15 if focus == "T/F" else 0.05
            f_boost = min(adjust_factor, emotional_sum / 5)
            final_dimension_scores["T/F"][0] -= f_boost  # 减少 T
            final_dimension_scores["T/F"][1] += f_boost  # 增加 F
        
        # J/P 维度 - 基于情绪稳定性
        if neutral > 0.5:
            adjust_factor = 0.15 if focus == "J/P" else 0.05
            j_boost = min(adjust_factor, (neutral - 0.5) / 2)
            final_dimension_scores["J/P"][0] += j_boost  # 增加 J
            final_dimension_scores["J/P"][1] -= j_boost  # 减少 P
    
    # 归一化评分
    for dimension in final_dimension_scores:
        scores = final_dimension_scores[dimension]
        total = sum(scores)
        if total > 0:
            final_dimension_scores[dimension] = [s/total for s in scores]
    
    # 确定最终 MBTI 类型
    mbti_type = ""
    mbti_type += "I" if final_dimension_scores["I/E"][0] > final_dimension_scores["I/E"][1] else "E"
    mbti_type += "S" if final_dimension_scores["S/N"][0] > final_dimension_scores["S/N"][1] else "N"
    mbti_type += "T" if final_dimension_scores["T/F"][0] > final_dimension_scores["T/F"][1] else "F"
    mbti_type += "J" if final_dimension_scores["J/P"][0] > final_dimension_scores["J/P"][1] else "P"
    
    return {
        'mbti_type': mbti_type,
        'dimension_scores': final_dimension_scores
    }

# ================ 保存结果 ================

def save_assessment_results(results):
    """保存评估结果到文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(RESULTS_DIR, f"mbti_assessment_{timestamp}.json")
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return file_path

# ================ Flask 路由 ================

@app.route('/')
def index():
    """渲染主页"""
    return render_template('index.html')

@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    """处理视频帧进行情绪分析"""
    try:
        # 接收前端发送的图像
        data = request.json
        image_data = data.get('image', '')
        
        # 处理 base64 图像
        if image_data:
            img = process_base64_image(image_data)
            
            # 分析情绪
            emotions, face_detected = analyze_emotion(img)
            
            return jsonify({
                'emotions': emotions, 
                'face_detected': face_detected
            })
        else:
            return jsonify({'error': '未收到图像数据'}), 400
            
    except Exception as e:
        logger.error(f"处理帧时出错: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/process_text', methods=['POST'])
def process_text():
    """处理文本进行 MBTI 分析"""
    try:
        # 接收前端发送的文本
        data = request.json
        text = data.get('text', '')
        
        if text:
            # 分析 MBTI
            mbti_results = analyze_mbti_text(text)
            
            return jsonify(mbti_results)
        else:
            return jsonify({'error': '未收到文本数据'}), 400
            
    except Exception as e:
        logger.error(f"处理文本时出错: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/process_speech', methods=['POST'])
def process_speech():
    """处理语音识别结果"""
    try:
        # 接收前端发送的语音识别文本
        data = request.json
        speech_text = data.get('speech_text', '')
        
        if speech_text:
            # 记录原始语音识别文本
            logger.info(f"收到语音识别文本: {speech_text}")
            
            # 使用与文本分析相同的方法分析MBTI
            mbti_results = analyze_mbti_text(speech_text)
            
            return jsonify({
                'original_text': speech_text,
                'mbti_results': mbti_results
            })
        else:
            return jsonify({'error': '未收到语音识别文本'}), 400
            
    except Exception as e:
        logger.error(f"处理语音识别结果时出错: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/final_results', methods=['POST'])
def generate_final_results():
    """生成最终的 MBTI 预测结果"""
    try:
        # 接收所有问题的回答和情绪数据
        data = request.json
        responses = data.get('responses', [])
        
        if responses:
            # 综合分析生成最终预测
            final_results = integrate_multimodal_data(responses)
            
            # 保存结果
            file_path = save_assessment_results({
                "timestamp": datetime.now().isoformat(),
                "responses": responses,
                "final_results": final_results
            })
            
            # 添加结果保存路径
            final_results['saved_to'] = file_path
            
            return jsonify(final_results)
        else:
            return jsonify({'error': '未收到回答数据'}), 400
            
    except Exception as e:
        logger.error(f"生成最终结果时出错: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/results/<path:filename>')
def download_results(filename):
    """允许下载保存的结果文件"""
    return send_from_directory(RESULTS_DIR, filename, as_attachment=True)

@app.route('/test')
def test_page():
    """测试页面，用于验证服务器运行正常"""
    return "MBTI 评估系统服务器运行正常！"

# ================ 应用启动 ================

def initialize_app():
    """初始化应用，加载所有模型"""
    logger.info("正在初始化 MBTI 评估系统...")
    
    # 初始化人脸检测器
    initialize_face_detector()
    
    # 加载情绪模型
    emotion_model_loaded = load_emotion_model()
    
    # 加载文本模型
    text_model_loaded = load_text_model()
    
    if emotion_model_loaded and text_model_loaded:
        logger.info("所有模型加载成功！系统准备就绪。")
    else:
        logger.warning("警告：一些模型未能成功加载。系统可能无法正常工作。")

if __name__ == '__main__':
    # 初始化应用
    initialize_app()
    
    # 启动服务器
    app.run(debug=True, host='0.0.0.0', port=5000)