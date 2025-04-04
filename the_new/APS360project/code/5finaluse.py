import os
import sys
import cv2
import time
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import threading
import speech_recognition as sr
from datetime import datetime

# Set paths to model files
BASE_DIR = r"C:\Users\lnasl\Desktop\APS360\APS360\Model\TrainedModel"
EMOTION_MODEL_PATH = os.path.join(BASE_DIR, "emotion", "improved_emotion_model.pth")
TEXT_MODEL_DIR = os.path.join(BASE_DIR, "text", "ml")

# Define emotion categories
EMOTION_CATEGORIES = ['Anger', 'Confusion', 'Contempt', 'Disgust', 
                      'Happiness', 'Neutral', 'Sadness', 'Surprise']

# Define MBTI dimensions
MBTI_DIMENSIONS = ["I/E", "S/N", "T/F", "J/P"]

# ================ CNN Emotion Recognition Model ================

class SpatialAttention(nn.Module):
    """Spatial attention module from the CNN model"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
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
    """Channel attention module from the CNN model"""
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
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class EnhancedEmotionModel(nn.Module):
    """CNN model for emotion recognition based on the provided architecture"""
    def __init__(self, num_classes=8, dropout_rates=[0.5, 0.4, 0.3], backbone='efficientnet_b3'):
        super(EnhancedEmotionModel, self).__init__()
        
        # Choose base model
        if backbone == 'efficientnet_b3':
            self.base_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
            last_channel = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Add attention module
        self.cbam = CBAM(channels=last_channel, reduction=16, kernel_size=7)
        
        # Global pooling after feature extraction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
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
        
        # Specialized head for difficult-to-classify categories
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
        # Feature extraction
        features = self.base_model.features(x)
        
        # Apply attention mechanism
        features = self.cbam(features)
        
        # Global pooling
        features = self.avg_pool(features)
        features = torch.flatten(features, 1)
        
        # Main classifier output
        main_output = self.classifier(features)
        
        # We only need the main output for inference
        return main_output, features

class EmotionRecognizer:
    """Class for detecting faces and recognizing emotions using the pre-trained CNN model"""
    def __init__(self, model_path=EMOTION_MODEL_PATH):
        # Initialize face detector from OpenCV
        cascade_path = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

# 检查是否成功加载分类器
        if self.face_cascade.empty():
            print("错误: 无法加载人脸级联分类器XML文件")
            print("请确保文件路径正确，并且文件存在")       
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load pre-trained emotion recognition model
        try:
            self.model = EnhancedEmotionModel(num_classes=len(EMOTION_CATEGORIES))
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            print(f"Loaded emotion model from {model_path}")
        except Exception as e:
            print(f"Error loading emotion model: {e}")
            print("Using a placeholder emotion detector instead")
            self.model_loaded = False
        
        # Define transformation for input images
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Emotion categories
        self.emotions = EMOTION_CATEGORIES
    
    def detect_emotion(self, frame):
        """
        Detect faces in the frame and recognize emotions
        Returns: Dict with emotion probabilities, frame with annotations
        """
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None, frame  # No face detected
        
        # For simplicity, just consider the largest face
        largest_face_idx = np.argmax([w*h for (x, y, w, h) in faces])
        x, y, w, h = faces[largest_face_idx]
        
        # Draw rectangle around the face
        frame_with_face = frame.copy()
        cv2.rectangle(frame_with_face, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Extract the face ROI
        face_roi = frame[y:y+h, x:x+w]
        
        # Recognize emotion
        if self.model_loaded:
            # Preprocess face for the model
            try:
                # Convert BGR to RGB
                face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                pil_img = Image.fromarray(face_roi_rgb)
                # Apply transforms
                input_tensor = self.transform(pil_img)
                input_tensor = input_tensor.unsqueeze(0).to(self.device)
                
                # Get emotion predictions
                with torch.no_grad():
                    predictions, _ = self.model(input_tensor)
                    probabilities = F.softmax(predictions, dim=1)[0].cpu().numpy()
                
                # Create emotion dictionary
                emotion_dict = {emotion: float(prob) for emotion, prob in zip(self.emotions, probabilities)}
                
                # Get dominant emotion
                dominant_emotion = self.emotions[np.argmax(probabilities)]
                
                # Display the emotion on the frame
                cv2.putText(frame_with_face, dominant_emotion, (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            except Exception as e:
                print(f"Error in emotion recognition: {e}")
                # Generate random probabilities for testing purposes
                emotion_dict = {emotion: 1/len(self.emotions) for emotion in self.emotions}
                cv2.putText(frame_with_face, "Error", (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            # Generate random probabilities for testing purposes
            rnd_probs = np.random.random(len(self.emotions))
            rnd_probs = rnd_probs / np.sum(rnd_probs)  # Normalize to sum to 1
            emotion_dict = {emotion: float(prob) for emotion, prob in zip(self.emotions, rnd_probs)}
            
            dominant_emotion = self.emotions[np.argmax(rnd_probs)]
            cv2.putText(frame_with_face, dominant_emotion, (x, y-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        return emotion_dict, frame_with_face


# ================ Text MBTI Prediction Model ================

def robust_text_preprocessing(text):
    """Text preprocessing function from the text model"""
    import re
    
    # Ensure text is string
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' url ', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' email ', text)
    
    # Replace numbers with 'number' token
    text = re.sub(r'\d+', ' number ', text)
    
    # Handle punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Simple word stemming
    words = text.split()
    processed_words = []
    
    # English stopwords
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
        # Skip stopwords
        if word in stop_words:
            continue
            
        # Skip very short words
        if len(word) <= 2:
            continue
            
        # Simple stemming rules
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
    
    # Rejoin words
    text = ' '.join(processed_words)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

class MBTITextAnalyzer:
    """Class for predicting MBTI based on text using the pre-trained model"""
    def __init__(self, model_dir=TEXT_MODEL_DIR):
        try:
            # Load model
            with open(os.path.join(model_dir, 'model.pkl'), 'rb') as f:
                self.model = pickle.load(f)
            
            # Load vectorizer
            with open(os.path.join(model_dir, 'vectorizer.pkl'), 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            # Load label encoder
            with open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            # Load config
            with open(os.path.join(model_dir, 'config.json'), 'r') as f:
                self.config = json.load(f)
                
            print(f"Loaded text model: {self.config.get('model_name', 'Unknown')}")
            self.model_loaded = True
            
        except Exception as e:
            print(f"Error loading MBTI text model: {e}")
            self.model_loaded = False
    
    def predict_mbti(self, text):
        """Predict MBTI based on text input"""
        # In case model failed to load, return a placeholder prediction
        if not self.model_loaded:
            return self._get_placeholder_prediction()
        
        # Preprocess the text
        processed_text = robust_text_preprocessing(text)
        
        # Convert to feature vector
        text_vector = self.vectorizer.transform([processed_text])
        
        # Make prediction
        if hasattr(self.model, 'predict_proba'):
            # Get prediction probabilities
            proba = self.model.predict_proba(text_vector)[0]
            prediction = proba.argmax()
            mbti_type = self.label_encoder.inverse_transform([prediction])[0]
            
            # Extract confidence scores for each dimension
            dimension_scores = self._extract_dimension_scores(proba)
            
            return {
                'mbti_type': mbti_type,
                'dimension_scores': dimension_scores,
                'confidence': float(proba[prediction])
            }
        else:
            # For models without probability support
            prediction = self.model.predict(text_vector)[0]
            mbti_type = self.label_encoder.inverse_transform([prediction])[0]
            
            # Create placeholder dimension scores
            dimension_scores = self._extract_dimension_scores_from_type(mbti_type)
            
            return {
                'mbti_type': mbti_type,
                'dimension_scores': dimension_scores,
                'confidence': 1.0
            }
    
    def _extract_dimension_scores(self, probabilities):
        """
        Extract dimension scores (I/E, S/N, T/F, J/P) from model probabilities
        This associates each MBTI type probability with its contribution to each dimension
        """
        # Initialize scores for each dimension
        ie_scores = [0.0, 0.0]  # [I, E]
        sn_scores = [0.0, 0.0]  # [S, N]
        tf_scores = [0.0, 0.0]  # [T, F]
        jp_scores = [0.0, 0.0]  # [J, P]
        
        # Iterate through all types and their probabilities
        for i, type_prob in enumerate(probabilities):
            mbti_type = self.label_encoder.inverse_transform([i])[0]
            
            # Add probability to the corresponding dimension score
            ie_scores[0 if mbti_type[0] == 'I' else 1] += type_prob
            sn_scores[0 if mbti_type[1] == 'S' else 1] += type_prob
            tf_scores[0 if mbti_type[2] == 'T' else 1] += type_prob
            jp_scores[0 if mbti_type[3] == 'J' else 1] += type_prob
        
        # Create dimension scores dictionary
        dimension_scores = {
            "I/E": (ie_scores[0], ie_scores[1]),  # (I, E)
            "S/N": (sn_scores[0], sn_scores[1]),  # (S, N)
            "T/F": (tf_scores[0], tf_scores[1]),  # (T, F)
            "J/P": (jp_scores[0], jp_scores[1])   # (J, P)
        }
        
        return dimension_scores
    
    def _extract_dimension_scores_from_type(self, mbti_type):
        """Create dimension scores from a single MBTI type (for models without probability support)"""
        dimension_scores = {
            "I/E": (0.8, 0.2) if mbti_type[0] == 'I' else (0.2, 0.8),
            "S/N": (0.8, 0.2) if mbti_type[1] == 'S' else (0.2, 0.8),
            "T/F": (0.8, 0.2) if mbti_type[2] == 'T' else (0.2, 0.8),
            "J/P": (0.8, 0.2) if mbti_type[3] == 'J' else (0.2, 0.8)
        }
        return dimension_scores
    
    def _get_placeholder_prediction(self):
        """Create a placeholder prediction when model is not available"""
        dimension_scores = {
            "I/E": (0.5, 0.5),
            "S/N": (0.5, 0.5),
            "T/F": (0.5, 0.5),
            "J/P": (0.5, 0.5)
        }
        
        return {
            'mbti_type': 'ISFJ',  # Most common type
            'dimension_scores': dimension_scores,
            'confidence': 0.25
        }


# ================ Speech-to-Text Component ================

class SpeechToText:
    """Class for capturing audio and converting it to text"""
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.text = ""
        self.is_recording = False
        self.recording_thread = None
    
    def start_recording(self, duration=30):
        """Start recording audio for the specified duration"""
        self.is_recording = True
        self.recording_thread = threading.Thread(target=self._record_audio, args=(duration,))
        self.recording_thread.daemon = True
        self.recording_thread.start()
    
    def stop_recording(self):
        """Stop recording audio"""
        self.is_recording = False
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join()
    
    def _record_audio(self, duration):
        """Record audio and convert to text"""
        self.text = ""
        try:
            with sr.Microphone() as source:
                print("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print(f"Recording for {duration} seconds...")
                audio = self.recognizer.listen(source, timeout=duration)
                print("Converting speech to text...")
                self.text = self.recognizer.recognize_google(audio)
                print(f"Recognized: {self.text}")
        except sr.WaitTimeoutError:
            print("No speech detected within the time limit")
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
        except Exception as e:
            print(f"Error in speech recognition: {e}")
        finally:
            self.is_recording = False


# ================ Emotion Feature Extraction ================

class EmotionFeatureExtractor:
    """Class for extracting features from emotion data"""
    def __init__(self):
        self.emotion_samples = []  # List to store emotion samples over time
        self.emotion_peaks = {}    # Dictionary to store emotion peaks
        
    def add_sample(self, emotion_dict):
        """Add an emotion sample to the time series"""
        if emotion_dict:
            timestamp = time.time()
            self.emotion_samples.append((timestamp, emotion_dict))
            
            # Check for emotion peaks
            self._check_for_peaks(emotion_dict, threshold=0.6)
    
    def _check_for_peaks(self, emotion_dict, threshold=0.6):
        """Check if any emotion exceeds the threshold to be considered a peak"""
        for emotion, prob in emotion_dict.items():
            if prob > threshold:
                if emotion not in self.emotion_peaks:
                    self.emotion_peaks[emotion] = []
                self.emotion_peaks[emotion].append(time.time())
    
    def get_emotion_distribution(self):
        """Calculate the distribution of emotions over the samples"""
        if not self.emotion_samples:
            return {emotion: 0.0 for emotion in EMOTION_CATEGORIES}
        
        # Initialize counters for each emotion
        emotion_counts = {}
        
        # Aggregate emotions across all samples
        for _, emotion_dict in self.emotion_samples:
            for emotion, prob in emotion_dict.items():
                if emotion not in emotion_counts:
                    emotion_counts[emotion] = 0.0
                emotion_counts[emotion] += prob
        
        # Normalize to get percentages
        total = sum(emotion_counts.values())
        if total > 0:
            emotion_distribution = {emotion: count/total for emotion, count in emotion_counts.items()}
        else:
            emotion_distribution = {emotion: 0.0 for emotion in emotion_counts}
        
        return emotion_distribution
    
    def get_emotion_peaks(self):
        """Get the peaks for each emotion"""
        return self.emotion_peaks
    
    def get_emotion_vector(self):
        """Get emotion distribution as a vector"""
        distribution = self.get_emotion_distribution()
        return [distribution.get(emotion, 0.0) for emotion in EMOTION_CATEGORIES]
    
    def clear(self):
        """Clear all emotion samples and peaks"""
        self.emotion_samples = []
        self.emotion_peaks = {}


# ================ Multimodal Integration ================

class MBTIMultimodalPredictor:
    """Class for integrating text and emotion features to predict MBTI"""
    def __init__(self):
        pass
    
    def predict(self, text_mbti, emotion_distribution, emotion_peaks=None, question_focus=None):
        """
        Integrate text MBTI prediction with emotion features
        
        Args:
            text_mbti: Dict with MBTI prediction from text model
            emotion_distribution: Dict with emotion distribution percentages
            emotion_peaks: Dict with emotion peak timestamps (optional)
            question_focus: Which MBTI dimension the question focuses on
            
        Returns:
            Dict with final MBTI dimension probabilities
        """
        # Extract dimension scores from text MBTI
        dimension_scores = text_mbti['dimension_scores']
        
        # Initialize final dimension scores with text prediction
        final_dimension_scores = {
            "I/E": list(dimension_scores["I/E"]),  # (I, E)
            "S/N": list(dimension_scores["S/N"]),  # (S, N)
            "T/F": list(dimension_scores["T/F"]),  # (T, F)
            "J/P": list(dimension_scores["J/P"])   # (J, P)
        }
        
        # Apply emotion-based adjustments
        # Extract emotion percentages
        happiness = emotion_distribution.get('Happiness', 0)
        surprise = emotion_distribution.get('Surprise', 0)
        confusion = emotion_distribution.get('Confusion', 0)
        neutral = emotion_distribution.get('Neutral', 0)
        sadness = emotion_distribution.get('Sadness', 0)
        anger = emotion_distribution.get('Anger', 0)
        contempt = emotion_distribution.get('Contempt', 0)
        disgust = emotion_distribution.get('Disgust', 0)
        
        # I/E dimension - adjust based on Happiness and Surprise
        # If Happiness and Surprise are high, slightly favor E
        if happiness + surprise > 0.4 and neutral < 0.4:
            # Smaller adjustment if not focused on I/E
            adjust_factor = 0.15 if question_focus == "I/E" else 0.05
            e_boost = min(adjust_factor, (happiness + surprise - 0.4) / 2)
            final_dimension_scores["I/E"][0] -= e_boost  # Decrease I
            final_dimension_scores["I/E"][1] += e_boost  # Increase E
        
        # S/N dimension - adjust based on Confusion
        # If Confusion is high when talking about abstract concepts, might favor S
        if confusion > 0.2:
            adjust_factor = 0.15 if question_focus == "S/N" else 0.05
            n_penalty = min(adjust_factor, confusion / 10)
            final_dimension_scores["S/N"][0] += n_penalty  # Increase S
            final_dimension_scores["S/N"][1] -= n_penalty  # Decrease N
        
        # T/F dimension - adjust based on emotional expressiveness
        # If showing more emotions, might slightly favor F
        emotional_sum = sadness + anger + happiness
        if emotional_sum > 0.3:
            adjust_factor = 0.15 if question_focus == "T/F" else 0.05
            f_boost = min(adjust_factor, emotional_sum / 5)
            final_dimension_scores["T/F"][0] -= f_boost  # Decrease T
            final_dimension_scores["T/F"][1] += f_boost  # Increase F
        
        # J/P dimension - based on emotional stability
        # More neutral = organized/structured = J
        # More variable emotions = adaptable/flexible = P
        if neutral > 0.5:
            adjust_factor = 0.15 if question_focus == "J/P" else 0.05
            j_boost = min(adjust_factor, (neutral - 0.5) / 2)
            final_dimension_scores["J/P"][0] += j_boost  # Increase J
            final_dimension_scores["J/P"][1] -= j_boost  # Decrease P
        
        # Normalize probabilities to ensure they sum to 1
        for dimension in final_dimension_scores:
            scores = final_dimension_scores[dimension]
            total = sum(scores)
            if total > 0:
                final_dimension_scores[dimension] = [s/total for s in scores]
        
        # Determine final MBTI type
        mbti_type = ""
        mbti_type += "I" if final_dimension_scores["I/E"][0] > final_dimension_scores["I/E"][1] else "E"
        mbti_type += "S" if final_dimension_scores["S/N"][0] > final_dimension_scores["S/N"][1] else "N"
        mbti_type += "T" if final_dimension_scores["T/F"][0] > final_dimension_scores["T/F"][1] else "F"
        mbti_type += "J" if final_dimension_scores["J/P"][0] > final_dimension_scores["J/P"][1] else "P"
        
        # Return final results
        return {
            'mbti_type': mbti_type,
            'dimension_scores': final_dimension_scores,
            'text_mbti': text_mbti['mbti_type'],
            'text_confidence': text_mbti['confidence']
        }


# ================ Main MBTI Assessment System ================

class MBTIAssessment:
    """Main class for running the MBTI assessment"""
    def __init__(self, output_dir=None):
        # Set output directory
        self.output_dir = output_dir or os.path.join(os.path.expanduser("~"), "MBTI_Results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self.emotion_recognizer = EmotionRecognizer()
        self.speech_to_text = SpeechToText()
        self.mbti_analyzer = MBTITextAnalyzer()
        self.emotion_extractor = EmotionFeatureExtractor()
        self.multimodal_predictor = MBTIMultimodalPredictor()
        
        # Store results for each question
        self.question_results = []
        
        # Questions for assessment
        self.questions = [
            "Tell me about your ideal social gathering. Do you prefer large parties with many people or smaller, intimate gatherings?",
            "When making decisions, do you tend to focus more on facts and details, or do you consider the bigger picture and possibilities?",
            "How do you approach deadlines and planning? Do you prefer having a structured schedule or keeping your options open?"
        ]
        
        # Question focus areas
        self.question_focus = [
            "I/E",  # Question 1 focuses on Introversion/Extraversion
            "S/N",  # Question 2 focuses on Sensing/Intuition
            "J/P"   # Question 3 focuses on Judging/Perceiving
        ]
        
    def run_assessment(self):
        """Run the complete MBTI assessment process"""
        print("\n===== MBTI PERSONALITY ASSESSMENT =====")
        print("This assessment will ask you 3 questions.")
        print("For each question, your facial expressions and speech will be analyzed.")
        print("Please face the camera and speak clearly.\n")
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        
        try:
            # For each question
            for i, question in enumerate(self.questions):
                print(f"\nQuestion {i+1}: {question}")
                print("Press ENTER when you're ready to answer...")
                input()
                
                # Clear previous emotion data
                self.emotion_extractor.clear()
                
                # Start recording audio
                self.speech_to_text.start_recording(duration=30)
                
                # Record start time
                start_time = time.time()
                max_duration = 30  # Maximum duration in seconds
                
                # Create window for displaying video
                cv2.namedWindow('MBTI Assessment', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('MBTI Assessment', 640, 480)
                
                # Capture video and emotions while recording
                while self.speech_to_text.is_recording and (time.time() - start_time) < max_duration:
                    ret, frame = cap.read()
                    if not ret:
                        print("Error capturing frame")
                        break
                    
                    # Detect emotion
                    emotion_dict, annotated_frame = self.emotion_recognizer.detect_emotion(frame)
                    
                    # Add emotion sample
                    if emotion_dict:
                        self.emotion_extractor.add_sample(emotion_dict)
                    
                    # Display frame
                    if annotated_frame is not None:
                        # Display recording indicator
                        remaining = max(0, max_duration - (time.time() - start_time))
                        cv2.putText(
                            annotated_frame,
                            f"Recording... {remaining:.1f}s remaining", 
                            (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, 
                            (0, 0, 255), 
                            2
                        )
                        cv2.imshow('MBTI Assessment', annotated_frame)
                    
                    # Break loop on 'q' key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.speech_to_text.stop_recording()
                        break
                
                # Make sure recording has stopped
                if self.speech_to_text.is_recording:
                    self.speech_to_text.stop_recording()
                
                # Get speech recognition result
                text_response = self.speech_to_text.text
                print(f"Your response: {text_response}")
                
                # Analyze text for MBTI
                text_mbti = self.mbti_analyzer.predict_mbti(text_response)
                
                # Get emotion distribution
                emotion_distribution = self.emotion_extractor.get_emotion_distribution()
                emotion_peaks = self.emotion_extractor.get_emotion_peaks()
                
                # Integrate text and emotion for MBTI prediction
                final_mbti = self.multimodal_predictor.predict(
                    text_mbti, 
                    emotion_distribution, 
                    emotion_peaks,
                    question_focus=self.question_focus[i]
                )
                
                # Store results for this question
                self.question_results.append({
                    "question": question,
                    "focus": self.question_focus[i],
                    "text_response": text_response,
                    "text_mbti": text_mbti,
                    "emotion_distribution": emotion_distribution,
                    "emotion_peaks": emotion_peaks,
                    "final_mbti": final_mbti
                })
                
                # Display interim results
                self._display_question_results(i)
                
                # Wait before proceeding to next question
                if i < len(self.questions) - 1:
                    print("\nPress ENTER to continue to the next question...")
                    input()
            
            # Generate final MBTI prediction
            final_result = self._generate_final_mbti()
            
            # Display final results
            self._display_final_results(final_result)
            
            # Save results
            self._save_results()
            
        finally:
            # Release resources
            cap.release()
            cv2.destroyAllWindows()
    
    def _display_question_results(self, question_idx):
        """Display results for a specific question"""
        result = self.question_results[question_idx]
        
        print("\n----- Question Analysis -----")
        print(f"Question {question_idx+1} (Focus: {result['focus']}): {result['question']}")
        print(f"Your response: {result['text_response']}")
        
        print("\nEmotion Distribution:")
        for emotion, percent in result['emotion_distribution'].items():
            print(f"  {emotion}: {percent*100:.1f}%")
        
        print("\nMBTI Prediction from text:")
        text_mbti = result['text_mbti']
        print(f"  Type: {text_mbti['mbti_type']} (Confidence: {text_mbti['confidence']*100:.1f}%)")
        
        print("\nDimension scores from text:")
        for dimension, scores in text_mbti['dimension_scores'].items():
            labels = dimension.split('/')
            print(f"  {dimension}: {labels[0]}={scores[0]*100:.1f}%, {labels[1]}={scores[1]*100:.1f}%")
        
        print("\nMBTI Prediction with emotion integration:")
        final_mbti = result['final_mbti']
        print(f"  Type: {final_mbti['mbti_type']}")
        
        print("\nFinal dimension scores:")
        for dimension, scores in final_mbti['dimension_scores'].items():
            labels = dimension.split('/')
            print(f"  {dimension}: {labels[0]}={scores[0]*100:.1f}%, {labels[1]}={scores[1]*100:.1f}%")
    
    def _generate_final_mbti(self):
        """Generate final MBTI prediction by aggregating all question results"""
        # Initialize dimension scores
        final_dimension_scores = {
            "I/E": [0, 0],
            "S/N": [0, 0],
            "T/F": [0, 0],
            "J/P": [0, 0]
        }
        
        # Combine results from all questions
        for result in self.question_results:
            focus = result['focus']
            for dimension, scores in result['final_mbti']['dimension_scores'].items():
                # Give more weight to questions that focus on this dimension
                weight = 2.0 if dimension == focus else 1.0
                final_dimension_scores[dimension][0] += scores[0] * weight
                final_dimension_scores[dimension][1] += scores[1] * weight
        
        # Normalize
        for dimension in final_dimension_scores:
            total = sum(final_dimension_scores[dimension])
            if total > 0:
                final_dimension_scores[dimension] = [s/total for s in final_dimension_scores[dimension]]
        
        # Determine final MBTI type
        mbti_type = ""
        mbti_type += "I" if final_dimension_scores["I/E"][0] > final_dimension_scores["I/E"][1] else "E"
        mbti_type += "S" if final_dimension_scores["S/N"][0] > final_dimension_scores["S/N"][1] else "N"
        mbti_type += "T" if final_dimension_scores["T/F"][0] > final_dimension_scores["T/F"][1] else "F"
        mbti_type += "J" if final_dimension_scores["J/P"][0] > final_dimension_scores["J/P"][1] else "P"
        
        return {
            "mbti_type": mbti_type,
            "dimension_scores": final_dimension_scores
        }
    
    def _display_final_results(self, final_result):
        """Display the final MBTI prediction results"""
        print("\n\n===== FINAL MBTI PREDICTION =====")
        print(f"Your MBTI Type: {final_result['mbti_type']}")
        print("\nDimension Probabilities:")
        
        for dimension, scores in final_result['dimension_scores'].items():
            labels = dimension.split('/')
            print(f"  {dimension}: {labels[0]}={scores[0]*100:.1f}%, {labels[1]}={scores[1]*100:.1f}%")
        
        # Create bar chart for visualization
        self._plot_mbti_results(final_result)
    
    def _plot_mbti_results(self, final_result):
        """Plot MBTI results as a bar chart"""
        dimensions = ["I/E", "S/N", "T/F", "J/P"]
        labels = ["Introversion/Extraversion", "Sensing/Intuition", 
                 "Thinking/Feeling", "Judging/Perceiving"]
        
        # Prepare data
        scores = [final_result['dimension_scores'][dim] for dim in dimensions]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot each dimension
        for i, (dimension, score, label) in enumerate(zip(dimensions, scores, labels)):
            dim_labels = dimension.split('/')
            
            # Create subplot
            plt.subplot(2, 2, i+1)
            plt.bar([dim_labels[0], dim_labels[1]], score, color=['#4285F4', '#34A853'])
            plt.title(label)
            plt.ylim(0, 1)
            
            # Add percentage text
            for j, s in enumerate(score):
                plt.text(j, s + 0.05, f"{s*100:.1f}%", ha='center')
        
        # Add overall title
        plt.suptitle(f"MBTI Type: {final_result['mbti_type']}", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(self.output_dir, f"mbti_results_{timestamp}.png")
        plt.savefig(file_path)
        
        print(f"\nResults chart saved to: {file_path}")
        
        # Show plot
        plt.show()
    
    def _save_results(self):
        """Save the assessment results to a file"""
        # Create a dictionary with all the results
        results = {
            "timestamp": datetime.now().isoformat(),
            "questions": self.questions,
            "question_results": []
        }
        
        # Format each question result
        for result in self.question_results:
            result_dict = {
                "question": result["question"],
                "focus": result["focus"],
                "text_response": result["text_response"],
                "emotion_distribution": {k: float(v) for k, v in result["emotion_distribution"].items()},
                "text_mbti_type": result["text_mbti"]["mbti_type"],
                "text_confidence": float(result["text_mbti"]["confidence"]),
                "final_mbti_type": result["final_mbti"]["mbti_type"],
                "final_dimension_scores": {
                    dim: [float(s) for s in scores] 
                    for dim, scores in result["final_mbti"]["dimension_scores"].items()
                }
            }
            results["question_results"].append(result_dict)
        
        # Add final result
        final_result = self._generate_final_mbti()
        results["final_result"] = {
            "mbti_type": final_result["mbti_type"],
            "dimension_scores": {
                dim: [float(s) for s in scores] 
                for dim, scores in final_result["dimension_scores"].items()
            }
        }
        
        # Save to JSON file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(self.output_dir, f"mbti_assessment_{timestamp}.json")
        
        with open(file_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {file_path}")


# ================ Main Function ================

def main():
    """Main function to run the assessment"""
    print("Starting MBTI Personality Assessment System...")
    
    # Create output directory
    output_dir = os.path.join(os.path.expanduser("~"), "MBTI_Results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create and run assessment
    assessment = MBTIAssessment(output_dir)
    assessment.run_assessment()

if __name__ == "__main__":
    main()