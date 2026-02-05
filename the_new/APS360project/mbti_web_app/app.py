"""
MBTI Assessment Flask Web Application
Main application file containing only Flask routes
"""

import uuid
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory
from datetime import datetime

# Import configuration
import config

# Import services
from services.model_loader import get_model_manager
from services.emotion_service import analyze_emotion
from services.text_service import (
    analyze_mbti_text,
    integrate_multimodal_data,
    save_assessment_results
)

# Import utilities
from utils.image_utils import process_base64_image

# Import database functions
from database import save_prediction, save_question_usage

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask application
app = Flask(__name__)

# Get global model manager
manager = get_model_manager()


# ================ Flask Routes ================

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    """
    Process video frame for emotion analysis

    Request body:
        {
            "image": "base64_encoded_image"
        }

    Returns:
        JSON: {
            "emotions": {...},
            "face_detected": bool
        }
    """
    try:
        # Receive image from frontend
        data = request.json
        image_data = data.get('image', '')

        # Process base64 image
        if image_data:
            img = process_base64_image(image_data)

            # Analyze emotion
            emotions, face_detected = analyze_emotion(img)

            return jsonify({
                'emotions': emotions,
                'face_detected': face_detected
            })
        else:
            return jsonify({'error': 'No image data received'}), 400

    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/process_text', methods=['POST'])
def process_text():
    """
    Process text for MBTI analysis

    Request body:
        {
            "text": "user_text"
        }

    Returns:
        JSON: {
            "mbti_type": str,
            "dimension_scores": {...},
            "confidence": float
        }
    """
    try:
        # Receive text from frontend
        data = request.json
        text = data.get('text', '')

        if text:
            # Analyze MBTI
            mbti_results = analyze_mbti_text(text)
            return jsonify(mbti_results)
        else:
            return jsonify({'error': 'No text data received'}), 400

    except Exception as e:
        logger.error(f"Error processing text: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/process_speech', methods=['POST'])
def process_speech():
    """
    Process speech recognition results

    Request body:
        {
            "speech_text": "recognized_text"
        }

    Returns:
        JSON: {
            "original_text": str,
            "mbti_results": {...}
        }
    """
    try:
        # Receive speech recognition text from frontend
        data = request.json
        speech_text = data.get('speech_text', '')

        if speech_text:
            # Log original speech recognition text
            logger.info(f"Received speech recognition text: {speech_text}")

            # Use same method as text analysis to analyze MBTI
            mbti_results = analyze_mbti_text(speech_text)

            return jsonify({
                'original_text': speech_text,
                'mbti_results': mbti_results
            })
        else:
            return jsonify({'error': 'No speech recognition text received'}), 400

    except Exception as e:
        logger.error(f"Error processing speech recognition results: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/final_results', methods=['POST'])
def generate_final_results():
    """
    Generate final MBTI prediction results

    Request body:
        {
            "responses": [
                {
                    "questionIndex": int,
                    "mbtiResults": {...},
                    "emotionData": [...]
                }
            ]
        }

    Returns:
        JSON: {
            "mbti_type": str,
            "dimension_scores": {...},
            "saved_to": str
        }
    """
    try:
        # Receive all question responses and emotion data
        data = request.json
        responses = data.get('responses', [])
        session_id = data.get('session_id', str(uuid.uuid4()))

        if responses:
            # Integrate analysis to generate final prediction
            final_results = integrate_multimodal_data(responses)

            # Generate prediction ID
            prediction_id = str(uuid.uuid4())

            # Save results to JSON file
            file_path = save_assessment_results({
                "prediction_id": prediction_id,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "responses": responses,
                "final_results": final_results
            })

            # Save to database
            try:
                # Prepare text responses for database
                text_responses = []
                emotion_data = []

                for idx, response in enumerate(responses):
                    # Extract text response
                    question_id = response.get('questionIndex', idx) + 1
                    text_content = response.get('text', '') or response.get('mbtiResults', {}).get('text', '')

                    if text_content:
                        text_responses.append({
                            "question_id": question_id,
                            "response": text_content
                        })

                    # Extract emotion data
                    emotion_points = response.get('emotionData', [])
                    for point in emotion_points:
                        emotion_data.append({
                            "timestamp": point.get('timestamp', ''),
                            "emotions": point.get('emotions', {}),
                            "question_index": idx
                        })

                # Save prediction to database
                save_prediction(
                    prediction_id=prediction_id,
                    session_id=session_id,
                    predicted_mbti=final_results['mbti_type'],
                    dimension_scores=final_results['dimension_scores'],
                    text_responses=text_responses,
                    emotion_data=emotion_data,
                    language=data.get('language', 'en'),
                    question_version='1.0',
                    model_version='1.0'
                )

                # Save question usage records
                for idx, response in enumerate(responses):
                    question_id = response.get('questionIndex', idx) + 1
                    text_content = response.get('text', '') or response.get('mbtiResults', {}).get('text', '')

                    if text_content:
                        usage_id = str(uuid.uuid4())
                        save_question_usage(
                            usage_id=usage_id,
                            prediction_id=prediction_id,
                            question_id=question_id,
                            question_order=idx + 1,
                            user_response_text=text_content
                        )

                logger.info(f"Saved prediction to database: {prediction_id} -> {final_results['mbti_type']}")

            except Exception as db_error:
                logger.warning(f"Database save failed (continuing anyway): {db_error}")
                # Don't fail the whole request if database save fails

            # Add result metadata
            final_results['prediction_id'] = prediction_id
            final_results['session_id'] = session_id
            final_results['saved_to'] = file_path

            return jsonify(final_results)
        else:
            return jsonify({'error': 'No response data received'}), 400

    except Exception as e:
        logger.error(f"Error generating final results: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/results/<path:filename>')
def download_results(filename):
    """Allow downloading saved result files"""
    return send_from_directory(config.RESULTS_DIR, filename, as_attachment=True)


# ================ Question Management API ================

@app.route('/api/questions', methods=['GET'])
def get_questions():
    """
    Get random questions for test session

    Query parameters:
        language (str): 'en' or 'zh', default 'en'
        count (int): Number of questions, default 3
        balanced (bool): Balance dimensions, default true

    Returns:
        JSON: {
            "questions": [...],
            "session_id": "...",
            "version": "1.0",
            "language": "..."
        }
    """
    try:
        # Get parameters
        language = request.args.get('language', 'en')
        count = int(request.args.get('count', 3))
        balanced = request.args.get('balanced', 'true').lower() == 'true'

        # Select question bank for language
        bank = manager.question_bank_zh if language == 'zh' else manager.question_bank_en

        if bank is None:
            logger.error(f"Question bank not initialized (language={language})")
            return jsonify({
                'error': 'Question bank not initialized',
                'language': language
            }), 500

        # Get random questions
        questions = bank.get_random_questions(count=count, balanced=balanced)

        # Generate session ID for tracking
        session_id = str(uuid.uuid4())

        logger.info(f"Generated {len(questions)} questions for session {session_id} (language={language}, balanced={balanced})")

        return jsonify({
            'questions': [q.to_dict() for q in questions],
            'session_id': session_id,
            'version': '1.0',
            'language': language,
            'count': len(questions)
        })

    except Exception as e:
        logger.error(f"Error getting questions: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/questions/<int:question_id>', methods=['GET'])
def get_question_by_id(question_id):
    """
    Get specific question by ID

    Path parameters:
        question_id (int): Question ID

    Query parameters:
        language (str): 'en' or 'zh', default 'en'

    Returns:
        JSON: Question details
    """
    try:
        language = request.args.get('language', 'en')
        bank = manager.question_bank_zh if language == 'zh' else manager.question_bank_en

        if bank is None:
            return jsonify({'error': 'Question bank not initialized'}), 500

        question = bank.get_question(question_id)

        if question:
            logger.info(f"Retrieved question ID={question_id} (language={language})")
            return jsonify(question.to_dict())
        else:
            return jsonify({'error': 'Question not found'}), 404

    except Exception as e:
        logger.error(f"Error getting question {question_id}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/questions/usage', methods=['POST'])
def record_question_usage():
    """
    Record which questions user answered

    Request body:
        {
            "prediction_id": "...",
            "session_id": "...",
            "questions": [
                {
                    "question_id": 1,
                    "question_order": 1,
                    "user_response": "..."
                }
            ]
        }

    Returns:
        JSON: {"status": "success", "recorded": count}
    """
    try:
        data = request.json

        if not data or 'questions' not in data:
            return jsonify({'error': 'Missing questions data'}), 400

        prediction_id = data.get('prediction_id', 'unknown')
        session_id = data.get('session_id', 'unknown')
        questions = data.get('questions', [])

        # Log (in production this should be saved to database)
        logger.info(f"Recording question usage: prediction_id={prediction_id}, session_id={session_id}, questions={len(questions)}")

        for q in questions:
            logger.debug(f"  Question {q.get('question_id')}: order={q.get('question_order')}, response_length={len(q.get('user_response', ''))}")

        # TODO: After database integration, save to question_usage table
        # Format:
        # INSERT INTO question_usage (usage_id, prediction_id, question_id, question_order, user_response_text, response_length)
        # VALUES (uuid, prediction_id, question_id, order, response, len(response))

        return jsonify({
            'status': 'success',
            'recorded': len(questions),
            'note': 'Currently logging only (database integration pending)'
        })

    except Exception as e:
        logger.error(f"Error recording question usage: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/questions/stats', methods=['GET'])
def get_question_stats():
    """
    Get question bank statistics

    Query parameters:
        language (str): 'en' or 'zh', default 'en'

    Returns:
        JSON: Question bank statistics
    """
    try:
        language = request.args.get('language', 'en')
        bank = manager.question_bank_zh if language == 'zh' else manager.question_bank_en

        if bank is None:
            return jsonify({'error': 'Question bank not initialized'}), 500

        # Statistics for each dimension
        stats = {
            'total_questions': len(bank.questions),
            'language': language,
            'by_dimension': {}
        }

        for dimension in config.MBTI_DIMENSIONS:
            dim_questions = bank.get_questions_by_dimension(dimension, language, active_only=True)
            stats['by_dimension'][dimension] = len(dim_questions)

        logger.info(f"Retrieved question stats (language={language}): total={stats['total_questions']}")

        return jsonify(stats)

    except Exception as e:
        logger.error(f"Error getting question stats: {e}")
        return jsonify({'error': str(e)}), 500


# ================ MBTI Description API ================

@app.route('/api/mbti/descriptions', methods=['GET'])
def get_mbti_descriptions():
    """
    Get all MBTI type descriptions

    Query parameters:
        language (str): 'en' or 'zh', default 'en'

    Returns:
        JSON: {
            "version": "1.0",
            "language": "en",
            "types": {
                "ISTJ": {...},
                "ISFJ": {...},
                ...
            }
        }
    """
    try:
        language = request.args.get('language', 'en')

        # Select descriptions for language
        descriptions = manager.mbti_descriptions_zh if language == 'zh' else manager.mbti_descriptions_en

        if descriptions is None or not descriptions.get('types'):
            logger.error(f"MBTI descriptions not initialized (language={language})")
            return jsonify({
                'error': 'MBTI descriptions not initialized',
                'language': language
            }), 500

        logger.info(f"Retrieved MBTI descriptions (language={language}, types={len(descriptions.get('types', {}))})")

        return jsonify(descriptions)

    except Exception as e:
        logger.error(f"Error getting MBTI descriptions: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/mbti/descriptions/<mbti_type>', methods=['GET'])
def get_mbti_description_by_type(mbti_type):
    """
    Get description for specific MBTI type

    Path parameters:
        mbti_type (str): MBTI type like 'INTJ', 'ENFP', etc.

    Query parameters:
        language (str): 'en' or 'zh', default 'en'

    Returns:
        JSON: {
            "type": "INTJ",
            "title": "The Mastermind",
            "description": "...",
            "characteristics": [...],
            "career_paths": "...",
            "development_suggestions": "..."
        }
    """
    try:
        language = request.args.get('language', 'en')
        mbti_type = mbti_type.upper()

        # Validate MBTI type format
        if len(mbti_type) != 4 or not all(c in 'IESTFNJP' for c in mbti_type):
            return jsonify({'error': 'Invalid MBTI type format'}), 400

        # Select descriptions for language
        descriptions = manager.mbti_descriptions_zh if language == 'zh' else manager.mbti_descriptions_en

        if descriptions is None:
            return jsonify({'error': 'MBTI descriptions not initialized'}), 500

        # Get description for specific type
        type_desc = descriptions.get('types', {}).get(mbti_type)

        if type_desc:
            logger.info(f"Retrieved MBTI description: {mbti_type} (language={language})")
            result = {
                'type': mbti_type,
                **type_desc
            }
            return jsonify(result)
        else:
            return jsonify({'error': f'MBTI type {mbti_type} not found'}), 404

    except Exception as e:
        logger.error(f"Error getting MBTI description {mbti_type}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/test')
def test_page():
    """Test page to verify server is running properly"""
    return "MBTI Assessment System server is running!"


# ================ Application Startup ================

if __name__ == '__main__':
    # Initialize application
    manager.initialize_all()

    # Start server
    app.run(debug=config.FLASK_DEBUG, host=config.FLASK_HOST, port=config.FLASK_PORT)
