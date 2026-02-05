"""
Text MBTI Analysis Service
Business logic for text-based MBTI prediction and multimodal integration
"""

import os
import json
import logging
from datetime import datetime

import config
from utils.text_utils import robust_text_preprocessing
from services.model_loader import get_model_manager

logger = logging.getLogger(__name__)


def analyze_mbti_text(text):
    """
    Analyze text to predict MBTI type

    Args:
        text: Input text string

    Returns:
        dict: {
            'mbti_type': str,
            'dimension_scores': dict,
            'confidence': float
        }
    """
    manager = get_model_manager()

    # Preprocess text
    processed_text = robust_text_preprocessing(text)

    # Transform to feature vector
    text_vector = manager.text_vectorizer.transform([processed_text])

    # Predict
    if hasattr(manager.text_model, 'predict_proba'):
        # Get prediction probabilities
        proba = manager.text_model.predict_proba(text_vector)[0]
        prediction = proba.argmax()
        mbti_type = manager.text_label_encoder.inverse_transform([prediction])[0]

        # Extract dimension scores
        dimension_scores = extract_dimension_scores(proba)

        return {
            'mbti_type': mbti_type,
            'dimension_scores': dimension_scores,
            'confidence': float(proba[prediction])
        }
    else:
        # Model doesn't support probabilities
        prediction = manager.text_model.predict(text_vector)[0]
        mbti_type = manager.text_label_encoder.inverse_transform([prediction])[0]

        # Create placeholder dimension scores
        dimension_scores = extract_dimension_scores_from_type(mbti_type)

        return {
            'mbti_type': mbti_type,
            'dimension_scores': dimension_scores,
            'confidence': 1.0
        }


def extract_dimension_scores(probabilities):
    """
    Extract dimension scores (I/E, S/N, T/F, J/P) from model probabilities

    Args:
        probabilities: Array of probabilities for all MBTI types

    Returns:
        dict: Dimension scores {
            "I/E": (I_score, E_score),
            "S/N": (S_score, N_score),
            "T/F": (T_score, F_score),
            "J/P": (J_score, P_score)
        }
    """
    manager = get_model_manager()

    # Initialize dimension scores
    ie_scores = [0.0, 0.0]  # [I, E]
    sn_scores = [0.0, 0.0]  # [S, N]
    tf_scores = [0.0, 0.0]  # [T, F]
    jp_scores = [0.0, 0.0]  # [J, P]

    # Iterate through all types and their probabilities
    for i, type_prob in enumerate(probabilities):
        mbti_type = manager.text_label_encoder.inverse_transform([i])[0]

        # Add probability to corresponding dimension scores
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


def extract_dimension_scores_from_type(mbti_type):
    """
    Create dimension scores from single MBTI type (for models without probabilities)

    Args:
        mbti_type: MBTI type string (e.g., "INTJ")

    Returns:
        dict: Dimension scores with simplified probabilities
    """
    dimension_scores = {
        "I/E": (0.8, 0.2) if mbti_type[0] == 'I' else (0.2, 0.8),
        "S/N": (0.8, 0.2) if mbti_type[1] == 'S' else (0.2, 0.8),
        "T/F": (0.8, 0.2) if mbti_type[2] == 'T' else (0.2, 0.8),
        "J/P": (0.8, 0.2) if mbti_type[3] == 'J' else (0.2, 0.8)
    }
    return dimension_scores


def integrate_multimodal_data(responses):
    """
    Integrate text and emotion data from all questions to predict final MBTI

    Args:
        responses: List of response dictionaries containing text and emotion data

    Returns:
        dict: {
            'mbti_type': str,
            'dimension_scores': dict
        }
    """
    # Question focus dimension mapping
    question_focus = [
        "I/E",  # Question 1 focuses on Introversion/Extraversion
        "S/N",  # Question 2 focuses on Sensing/Intuition
        "J/P"   # Question 3 focuses on Judging/Perceiving
    ]

    # Initialize final dimension scores
    final_dimension_scores = {
        "I/E": [0, 0],
        "S/N": [0, 0],
        "T/F": [0, 0],
        "J/P": [0, 0]
    }

    # Process each question's results
    for response in responses:
        q_idx = response.get('questionIndex', 0)

        # Get text MBTI analysis results
        text_mbti = response.get('mbtiResults', {})
        dimension_scores = text_mbti.get('dimension_scores', {})

        # Get emotion distribution
        emotion_data = response.get('emotionData', [])

        # Calculate average emotion distribution
        emotion_sum = {}
        for data_point in emotion_data:
            emotions = data_point.get('emotions', {})
            for emotion, value in emotions.items():
                emotion_sum[emotion] = emotion_sum.get(emotion, 0) + value

        # Normalize emotion distribution
        total = sum(emotion_sum.values()) if emotion_sum else 1
        emotion_distribution = {e: v / total for e, v in emotion_sum.items()}

        # Apply text-based scores to all dimensions
        focus = question_focus[q_idx] if q_idx < len(question_focus) else None

        for dimension, scores in dimension_scores.items():
            # Give higher weight to the focus dimension for this question
            weight = 2.0 if dimension == focus else 1.0
            if isinstance(scores, (list, tuple)) and len(scores) == 2:
                final_dimension_scores[dimension][0] += scores[0] * weight
                final_dimension_scores[dimension][1] += scores[1] * weight

        # Apply emotion adjustments
        # Extract emotion percentages
        happiness = emotion_distribution.get('Happiness', 0)
        surprise = emotion_distribution.get('Surprise', 0)
        confusion = emotion_distribution.get('Confusion', 0)
        neutral = emotion_distribution.get('Neutral', 0)
        sadness = emotion_distribution.get('Sadness', 0)
        anger = emotion_distribution.get('Anger', 0)

        # I/E dimension - adjust based on Happiness and Surprise
        if happiness + surprise > 0.4 and neutral < 0.4:
            # Larger adjustment if question focuses on I/E
            adjust_factor = 0.15 if focus == "I/E" else 0.05
            e_boost = min(adjust_factor, (happiness + surprise - 0.4) / 2)
            final_dimension_scores["I/E"][0] -= e_boost  # Decrease I
            final_dimension_scores["I/E"][1] += e_boost  # Increase E

        # S/N dimension - adjust based on Confusion
        if confusion > 0.2:
            adjust_factor = 0.15 if focus == "S/N" else 0.05
            n_penalty = min(adjust_factor, confusion / 10)
            final_dimension_scores["S/N"][0] += n_penalty  # Increase S
            final_dimension_scores["S/N"][1] -= n_penalty  # Decrease N

        # T/F dimension - adjust based on emotional expression
        emotional_sum = sadness + anger + happiness
        if emotional_sum > 0.3:
            adjust_factor = 0.15 if focus == "T/F" else 0.05
            f_boost = min(adjust_factor, emotional_sum / 5)
            final_dimension_scores["T/F"][0] -= f_boost  # Decrease T
            final_dimension_scores["T/F"][1] += f_boost  # Increase F

        # J/P dimension - adjust based on emotional stability
        if neutral > 0.5:
            adjust_factor = 0.15 if focus == "J/P" else 0.05
            j_boost = min(adjust_factor, (neutral - 0.5) / 2)
            final_dimension_scores["J/P"][0] += j_boost  # Increase J
            final_dimension_scores["J/P"][1] -= j_boost  # Decrease P

    # Normalize scores
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

    return {
        'mbti_type': mbti_type,
        'dimension_scores': final_dimension_scores
    }


def save_assessment_results(results):
    """
    Save assessment results to JSON file

    Args:
        results: Results dictionary to save

    Returns:
        str: File path where results were saved
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(config.RESULTS_DIR, f"mbti_assessment_{timestamp}.json")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return file_path
