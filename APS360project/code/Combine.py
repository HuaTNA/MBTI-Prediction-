import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import json
import os
from PIL import Image
from torchvision import transforms

class SpatialAttention(nn.Module):
    """
    Spatial attention module for the CBAM attention mechanism.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Create spatial attention map
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        x_out = self.conv1(x_cat)
        attention = self.sigmoid(x_out)
        
        return x * attention

class ChannelAttention(nn.Module):
    """
    Channel attention module for the CBAM attention mechanism.
    """
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
    """
    Convolutional Block Attention Module (CBAM).
    """
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class EmotionCNN(nn.Module):
    """
    Enhanced Emotion Recognition model that matches the architecture of the trained model.
    This model uses EfficientNet-B3 as backbone with CBAM attention mechanism.
    """
    def __init__(self, num_emotions=8):
        super(EmotionCNN, self).__init__()
        
        # Import EfficientNet-B3 from torchvision
        from torchvision import models
        
        # Use EfficientNet-B3 as the base model
        self.base_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        last_channel = self.base_model.classifier[1].in_features
        
        # Remove original classification head
        self.base_model.classifier = nn.Identity()
        
        # Add attention module
        self.cbam = CBAM(channels=last_channel, reduction=16, kernel_size=7)
        
        # Global pooling after feature extraction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(last_channel, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, num_emotions)
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
        
        # For compatibility with integration system, only return main output
        return main_output, features

class MBTIIntegrationSystem:
    def __init__(self, text_model_path, emotion_model_path):
        """
        Initialize the MBTI integration system with paths to both models.
        
        Args:
            text_model_path: Path to the text analysis model directory
            emotion_model_path: Path to the emotion recognition model directory
        """
        self.text_model_path = text_model_path
        self.emotion_model_path = emotion_model_path
        
        # MBTI dimensions and types
        self.dimensions = ['E-I', 'S-N', 'T-F', 'J-P']
        self.mbti_types = [
            'ISTJ', 'ISFJ', 'INFJ', 'INTJ', 
            'ISTP', 'ISFP', 'INFP', 'INTP',
            'ESTP', 'ESFP', 'ENFP', 'ENTP',
            'ESTJ', 'ESFJ', 'ENFJ', 'ENTJ'
        ]
        
        # Load text model components
        self.load_text_model()
        
        # Load emotion CNN model
        self.load_emotion_model()
        
        # Integration weights (how much each model contributes to final prediction)
        # These values can be tuned based on model performance
        self.text_weight = 0.6
        self.emotion_weight = 0.4
    
    def load_text_model(self):
        """Load text model components from pkl files"""
        try:
            # Load the text vectorizer
            vectorizer_path = os.path.join(self.text_model_path, 'vectorizer.pkl')
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            # Load the text model
            model_path = os.path.join(self.text_model_path, 'model.pkl')
            with open(model_path, 'rb') as f:
                self.text_model = pickle.load(f)
            
            # Load the label encoder
            encoder_path = os.path.join(self.text_model_path, 'label_encoder.pkl')
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            # Load config if available
            config_path = os.path.join(self.text_model_path, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.text_config = json.load(f)
                    
            print("Text model components loaded successfully.")
        except Exception as e:
            print(f"Error loading text model: {e}")
            raise
    
    def load_emotion_model(self):
        """Load emotion CNN model from .pth file"""
        try:
            # Determine the model file path
            model_files = [f for f in os.listdir(self.emotion_model_path) if f.endswith('.pth')]
            if not model_files:
                raise FileNotFoundError("No .pth emotion model found in the specified directory")
            
            model_path = os.path.join(self.emotion_model_path, model_files[0])
            print(f"Loading emotion model from: {model_path}")
            
            # Use our updated EmotionCNN class that matches the architecture of the trained model
            # The number of emotion categories is 8 based on the training code
            self.emotion_model = EmotionCNN(num_emotions=8)
            
            # Load model weights with strict=False to allow for potential minor differences
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            
            # Check if the state dict keys have 'module.' prefix (from DataParallel)
            if any(k.startswith('module.') for k in state_dict.keys()):
                # Remove the 'module.' prefix
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                
            # Load the state dict with strict=False to ignore potential minor differences
            self.emotion_model.load_state_dict(state_dict, strict=False)
            self.emotion_model.eval()
            
            # Define image transformation for preprocessing - matching the validation transform in training
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            print("Emotion model loaded successfully.")
        except Exception as e:
            print(f"Error loading emotion model: {e}")
            raise
    
    def process_text(self, text):
        """
        Process text input and predict MBTI dimensions.
        
        Args:
            text: Input text for personality analysis
            
        Returns:
            Dictionary containing predicted MBTI dimensions probabilities
        """
        # Vectorize the text input
        text_features = self.vectorizer.transform([text])
        
        # Check if the model supports predict_proba (e.g., LogisticRegression)
        if hasattr(self.text_model, 'predict_proba') and callable(getattr(self.text_model, 'predict_proba')):
            # Get prediction probabilities directly
            prediction_probs = self.text_model.predict_proba(text_features)
        elif hasattr(self.text_model, 'decision_function') and callable(getattr(self.text_model, 'decision_function')):
            # For LinearSVC, use decision_function and convert to pseudo-probabilities
            decisions = self.text_model.decision_function(text_features)
            
            # If decisions is a single score (binary classification)
            if decisions.ndim == 1:
                # Convert to binary pseudo-probabilities using sigmoid function
                def sigmoid(x):
                    return 1 / (1 + np.exp(-x))
                
                pos_probs = sigmoid(decisions)
                prediction_probs = np.column_stack((1 - pos_probs, pos_probs))
            else:
                # For multiclass, normalize decision values with softmax
                def softmax(x):
                    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
                    return e_x / e_x.sum(axis=1, keepdims=True)
                
                prediction_probs = softmax(decisions)
        else:
            # Fallback to hard predictions
            hard_predictions = self.text_model.predict(text_features)
            n_classes = len(self.mbti_types)
            
            # Convert hard predictions to one-hot encoded format
            prediction_probs = np.zeros((len(hard_predictions), n_classes))
            for i, pred in enumerate(hard_predictions):
                class_idx = self.label_encoder.transform([pred])[0]
                prediction_probs[i, class_idx] = 1.0
        
        # Map predictions to MBTI dimensions
        dimension_probabilities = {}
        
        # This assumes the model was trained to predict the full MBTI type
        # An alternative approach would be having separate models for each dimension
        # Mapping will depend on how the text model was trained
        
        # Determine the most likely MBTI type from text
        predicted_idx = prediction_probs.argmax(axis=1)[0]
        try:
            predicted_type = self.label_encoder.inverse_transform([predicted_idx])[0]
            
            # Convert to dimension probabilities
            # E-I dimension
            dimension_probabilities['E'] = sum(prediction_probs[0][self.label_encoder.transform([t])[0]] 
                                            for t in self.mbti_types if t[0] == 'E')
            dimension_probabilities['I'] = 1 - dimension_probabilities['E']
            
            # S-N dimension
            dimension_probabilities['S'] = sum(prediction_probs[0][self.label_encoder.transform([t])[0]] 
                                            for t in self.mbti_types if t[1] == 'S')
            dimension_probabilities['N'] = 1 - dimension_probabilities['S']
            
            # T-F dimension
            dimension_probabilities['T'] = sum(prediction_probs[0][self.label_encoder.transform([t])[0]] 
                                            for t in self.mbti_types if t[2] == 'T')
            dimension_probabilities['F'] = 1 - dimension_probabilities['T']
            
            # J-P dimension
            dimension_probabilities['J'] = sum(prediction_probs[0][self.label_encoder.transform([t])[0]] 
                                            for t in self.mbti_types if t[3] == 'J')
            dimension_probabilities['P'] = 1 - dimension_probabilities['J']
            
        except:
            # In case of any error with the label encoder or indexing, fall back to a balanced prediction
            print("Warning: Error in label encoding. Using balanced prediction for dimensions.")
            dimension_probabilities = {
                'E': 0.5, 'I': 0.5,
                'S': 0.5, 'N': 0.5,
                'T': 0.5, 'F': 0.5,
                'J': 0.5, 'P': 0.5
            }
        
        return dimension_probabilities
    
    def process_face_image(self, image_path):
        """
        Process facial image and predict MBTI dimensions based on emotions.
        
        Args:
            image_path: Path to the facial image
            
        Returns:
            Dictionary containing emotion-based MBTI dimension probabilities
        """
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.image_transform(image).unsqueeze(0)
        
        # Get emotion predictions - note the model now returns a tuple
        with torch.no_grad():
            emotion_output, _ = self.emotion_model(image_tensor)
            emotion_probs = F.softmax(emotion_output, dim=1).squeeze().numpy()
        
        # Map emotions to MBTI dimensions
        # Updated mapping based on the 8 emotion categories from your training code:
        # 0=Anger, 1=Confusion, 2=Contempt, 3=Disgust, 4=Happiness, 5=Neutral, 6=Sadness, 7=Surprise
        
        dimension_probabilities = {}
        
        # E (extroverted) correlates with Happiness, Surprise
        # I (introverted) correlates with Neutral, Sadness, Contempt, Confusion
        dimension_probabilities['E'] = emotion_probs[4] * 0.8 + emotion_probs[7] * 0.6
        dimension_probabilities['I'] = (emotion_probs[5] * 0.7 + emotion_probs[6] * 0.6 + 
                                       emotion_probs[2] * 0.5 + emotion_probs[1] * 0.5)
        
        # Normalize E-I
        total = dimension_probabilities['E'] + dimension_probabilities['I']
        dimension_probabilities['E'] /= total
        dimension_probabilities['I'] /= total
        
        # S (sensing) correlates with Neutral, Anger, Contempt
        # N (intuition) correlates with Surprise, Confusion
        dimension_probabilities['S'] = emotion_probs[5] * 0.7 + emotion_probs[0] * 0.6 + emotion_probs[2] * 0.5
        dimension_probabilities['N'] = emotion_probs[7] * 0.7 + emotion_probs[1] * 0.6
        
        # Normalize S-N
        total = dimension_probabilities['S'] + dimension_probabilities['N']
        dimension_probabilities['S'] /= total
        dimension_probabilities['N'] /= total
        
        # T (thinking) correlates with Neutral, Anger, Contempt
        # F (feeling) correlates with Happiness, Sadness, Confusion, Disgust
        dimension_probabilities['T'] = emotion_probs[5] * 0.7 + emotion_probs[0] * 0.6 + emotion_probs[2] * 0.6
        dimension_probabilities['F'] = (emotion_probs[4] * 0.7 + emotion_probs[6] * 0.7 + 
                                      emotion_probs[1] * 0.5 + emotion_probs[3] * 0.5)
        
        # Normalize T-F
        total = dimension_probabilities['T'] + dimension_probabilities['F']
        dimension_probabilities['T'] /= total
        dimension_probabilities['F'] /= total
        
        # J (judging) correlates with Anger, Disgust, Contempt, Neutral
        # P (perceiving) correlates with Surprise, Happiness, Confusion
        dimension_probabilities['J'] = (emotion_probs[0] * 0.7 + emotion_probs[3] * 0.6 + 
                                      emotion_probs[2] * 0.7 + emotion_probs[5] * 0.5)
        dimension_probabilities['P'] = emotion_probs[7] * 0.7 + emotion_probs[4] * 0.6 + emotion_probs[1] * 0.5
        
        # Normalize J-P
        total = dimension_probabilities['J'] + dimension_probabilities['P']
        dimension_probabilities['J'] /= total
        dimension_probabilities['P'] /= total
        
        return dimension_probabilities
    
    def integrate_predictions(self, text_probs, emotion_probs):
        """
        Integrate text and emotion predictions into final MBTI dimensions.
        
        Args:
            text_probs: MBTI dimension probabilities from text model
            emotion_probs: MBTI dimension probabilities from emotion model
            
        Returns:
            Final MBTI type and confidence scores
        """
        # Weighted integration of probabilities
        final_probs = {}
        
        for dim in ['E', 'I', 'S', 'N', 'T', 'F', 'J', 'P']:
            final_probs[dim] = (text_probs[dim] * self.text_weight + 
                               emotion_probs[dim] * self.emotion_weight)
        
        # Determine final MBTI type based on higher probability in each dimension
        mbti_result = ""
        confidence_scores = {}
        
        # E-I dimension
        if final_probs['E'] > final_probs['I']:
            mbti_result += "E"
            confidence_scores['E-I'] = final_probs['E']
        else:
            mbti_result += "I"
            confidence_scores['E-I'] = final_probs['I']
        
        # S-N dimension
        if final_probs['S'] > final_probs['N']:
            mbti_result += "S"
            confidence_scores['S-N'] = final_probs['S']
        else:
            mbti_result += "N"
            confidence_scores['S-N'] = final_probs['N']
        
        # T-F dimension
        if final_probs['T'] > final_probs['F']:
            mbti_result += "T"
            confidence_scores['T-F'] = final_probs['T']
        else:
            mbti_result += "F"
            confidence_scores['T-F'] = final_probs['F']
        
        # J-P dimension
        if final_probs['J'] > final_probs['P']:
            mbti_result += "J"
            confidence_scores['J-P'] = final_probs['J']
        else:
            mbti_result += "P"
            confidence_scores['J-P'] = final_probs['P']
        
        return mbti_result, confidence_scores, final_probs
    
    def predict(self, text, face_image_path):
        """
        Predict MBTI type using both text and facial image.
        
        Args:
            text: Input text for analysis
            face_image_path: Path to facial image
            
        Returns:
            Predicted MBTI type, confidence scores, and detailed probabilities
        """
        # Process text input
        text_probs = self.process_text(text)
        
        # Process facial image
        emotion_probs = self.process_face_image(face_image_path)
        
        # Integrate predictions
        mbti_type, confidence, all_probs = self.integrate_predictions(text_probs, emotion_probs)
        
        return {
            'mbti_type': mbti_type,
            'confidence': confidence,
            'text_probabilities': text_probs,
            'emotion_probabilities': emotion_probs,
            'integrated_probabilities': all_probs
        }

# Helper function to determine MBTI type from probabilities
def determine_mbti_type(probs):
    mbti_result = ""
    
    # E-I dimension
    if probs['E'] > probs['I']:
        mbti_result += "E"
    else:
        mbti_result += "I"
    
    # S-N dimension
    if probs['S'] > probs['N']:
        mbti_result += "S"
    else:
        mbti_result += "N"
    
    # T-F dimension
    if probs['T'] > probs['F']:
        mbti_result += "T"
    else:
        mbti_result += "F"
    
    # J-P dimension
    if probs['J'] > probs['P']:
        mbti_result += "J"
    else:
        mbti_result += "P"
    
    return mbti_result

# Main function to demonstrate system
def main():
    # Model paths
    text_model_path = r"C:\Users\lnasl\Desktop\APS360\APS360\Model\TrainedModel\text\ml"
    emotion_model_path = r"C:\Users\lnasl\Desktop\APS360\APS360\Model\TrainedModel\emotion"
    
    # Initialize the integration system
    mbti_system = MBTIIntegrationSystem(text_model_path, emotion_model_path)
    
    # Example prediction
    sample_text = "I enjoy spending time with friends and participating in group activities. I prefer concrete facts over abstract theories and make decisions based on logical analysis rather than feelings."
    
    # Let's use a path that is more likely to exist - either specify a real image path or use a placeholder path
    sample_image_path = r"C:\Users\lnasl\Desktop\APS360\APS360\Data\example_face.jpg"  # 请替换为实际图像路径
    
    # Check if the image path exists
    if not os.path.exists(sample_image_path):
        print(f"Warning: Example image path does not exist: {sample_image_path}")
        print("Using text-only prediction as fallback.")
        
        # Get text prediction only
        text_probs = mbti_system.process_text(sample_text)
        
        # Calculate confidence scores correctly from individual dimension probabilities
        confidence_scores = {
            'E-I': max(text_probs['E'], text_probs['I']),
            'S-N': max(text_probs['S'], text_probs['N']),
            'T-F': max(text_probs['T'], text_probs['F']),
            'J-P': max(text_probs['J'], text_probs['P'])
        }
        
        result = {
            'mbti_type': determine_mbti_type(text_probs),
            'confidence': confidence_scores,
            'text_probabilities': text_probs,
            'emotion_probabilities': {},
            'integrated_probabilities': text_probs
        }
    else:
        # Get full prediction
        result = mbti_system.predict(sample_text, sample_image_path)
    
    # Print results
    print(f"Predicted MBTI Type: {result['mbti_type']}")
    print("\nConfidence Scores:")
    for dim, score in result['confidence'].items():
        print(f"{dim}: {score:.4f}")
    
    print("\nDetailed Probabilities:")
    print("Text Model:")
    for dim, prob in result['text_probabilities'].items():
        print(f"{dim}: {prob:.4f}")
    
    if result['emotion_probabilities']:
        print("\nEmotion Model:")
        for dim, prob in result['emotion_probabilities'].items():
            print(f"{dim}: {prob:.4f}")
    
    print("\nIntegrated Probabilities:")
    for dim, prob in result['integrated_probabilities'].items():
        print(f"{dim}: {prob:.4f}")

# Add main entry point check
if __name__ == "__main__":
    main()