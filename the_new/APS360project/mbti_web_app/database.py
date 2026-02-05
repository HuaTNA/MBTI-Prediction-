"""
PostgreSQL Database Management Module
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import JSON
import logging

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Database configuration
# Use environment variable for production, fallback to default for development
# Set DATABASE_URL environment variable or create a .env file
DATABASE_URL = os.getenv(
    'DATABASE_URL',
    'postgresql://postgres:password@localhost:5432/mbti_predictions'
)

# Create SQLAlchemy engine and session
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ===================== Data Model Definitions =====================

class PredictionData(Base):
    """Prediction data table"""
    __tablename__ = 'prediction_data'

    prediction_id = Column(String(36), primary_key=True)  # UUID
    session_id = Column(String(36), index=True)  # Associated with question session
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Prediction results
    predicted_mbti = Column(String(4), index=True)

    # Dimension scores (JSON)
    dimension_scores = Column(JSON)  # {"I": 0.6, "E": 0.4, "S": 0.7, ...}

    # Text input
    text_responses = Column(JSON)  # [{"question_id": 1, "response": "..."}]

    # Emotion data (JSON)
    emotion_data = Column(JSON)  # [{"timestamp": "...", "emotion": "happy", "confidence": 0.8}]

    # Metadata
    language = Column(String(10), default='en')
    question_version = Column(String(20))
    model_version = Column(String(20))

    # Relationships
    question_usages = relationship("QuestionUsage", back_populates="prediction")


class QuestionUsage(Base):
    """Question usage record table"""
    __tablename__ = 'question_usage'

    usage_id = Column(String(36), primary_key=True)
    prediction_id = Column(String(36), ForeignKey('prediction_data.prediction_id'))
    question_id = Column(Integer, index=True)
    question_order = Column(Integer)  # Which question number

    # User response
    user_response_text = Column(Text)
    response_length = Column(Integer)

    # Relationships
    prediction = relationship("PredictionData", back_populates="question_usages")


class UserSession(Base):
    """User session table"""
    __tablename__ = 'user_sessions'

    session_id = Column(String(36), primary_key=True)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)

    # Session information
    user_agent = Column(String(200))
    ip_address = Column(String(45))  # IPv6 support

    # Completion status
    completed = Column(Boolean, default=False)


# ===================== Database Initialization =====================

def init_database():
    """Initialize database and create all tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✓ Database tables created successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Database initialization failed: {e}")
        return False


# ===================== Database Operation Functions =====================

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_prediction(
    prediction_id: str,
    session_id: str,
    predicted_mbti: str,
    dimension_scores: dict,
    text_responses: list,
    emotion_data: list = None,
    language: str = 'en',
    question_version: str = '1.0',
    model_version: str = '1.0'
):
    """
    Save prediction results to database

    Args:
        prediction_id: Prediction ID (UUID)
        session_id: Session ID
        predicted_mbti: Predicted MBTI type
        dimension_scores: Dimension scores {"I": 0.6, "E": 0.4, ...}
        text_responses: Text responses [{"question_id": 1, "response": "..."}]
        emotion_data: Emotion data (optional)
        language: Language
        question_version: Question version
        model_version: Model version
    """
    db = SessionLocal()
    try:
        prediction = PredictionData(
            prediction_id=prediction_id,
            session_id=session_id,
            predicted_mbti=predicted_mbti,
            dimension_scores=dimension_scores,
            text_responses=text_responses,
            emotion_data=emotion_data or [],
            language=language,
            question_version=question_version,
            model_version=model_version
        )

        db.add(prediction)
        db.commit()

        logger.info(f"✓ Saved prediction result: {prediction_id} -> {predicted_mbti}")
        return True

    except Exception as e:
        db.rollback()
        logger.error(f"✗ Failed to save prediction: {e}")
        return False
    finally:
        db.close()


def save_question_usage(
    usage_id: str,
    prediction_id: str,
    question_id: int,
    question_order: int,
    user_response_text: str
):
    """
    Save question usage record

    Args:
        usage_id: Usage record ID (UUID)
        prediction_id: Associated prediction ID
        question_id: Question ID
        question_order: Question order (which number)
        user_response_text: User response text
    """
    db = SessionLocal()
    try:
        usage = QuestionUsage(
            usage_id=usage_id,
            prediction_id=prediction_id,
            question_id=question_id,
            question_order=question_order,
            user_response_text=user_response_text,
            response_length=len(user_response_text)
        )

        db.add(usage)
        db.commit()

        logger.info(f"✓ Saved question usage: Q{question_id} for {prediction_id}")
        return True

    except Exception as e:
        db.rollback()
        logger.error(f"✗ Failed to save question usage: {e}")
        return False
    finally:
        db.close()


def create_user_session(
    session_id: str,
    user_agent: str = None,
    ip_address: str = None
):
    """Create user session record"""
    db = SessionLocal()
    try:
        session = UserSession(
            session_id=session_id,
            user_agent=user_agent,
            ip_address=ip_address
        )

        db.add(session)
        db.commit()

        logger.info(f"✓ Created session: {session_id}")
        return True

    except Exception as e:
        db.rollback()
        logger.error(f"✗ Failed to create session: {e}")
        return False
    finally:
        db.close()


def complete_user_session(session_id: str):
    """Mark session as completed"""
    db = SessionLocal()
    try:
        session = db.query(UserSession).filter_by(session_id=session_id).first()
        if session:
            session.completed = True
            session.end_time = datetime.utcnow()
            db.commit()
            logger.info(f"✓ Session completed: {session_id}")
            return True
        return False

    except Exception as e:
        db.rollback()
        logger.error(f"✗ Failed to update session: {e}")
        return False
    finally:
        db.close()


# ===================== Statistics Query Functions =====================

def get_prediction_statistics():
    """Get prediction statistics"""
    db = SessionLocal()
    try:
        total_predictions = db.query(PredictionData).count()

        # Count of each MBTI type
        mbti_counts = {}
        for mbti_type in ['INTJ', 'INTP', 'ENTJ', 'ENTP',
                          'INFJ', 'INFP', 'ENFJ', 'ENFP',
                          'ISTJ', 'ISFJ', 'ESTJ', 'ESFJ',
                          'ISTP', 'ISFP', 'ESTP', 'ESFP']:
            count = db.query(PredictionData).filter_by(predicted_mbti=mbti_type).count()
            mbti_counts[mbti_type] = count

        return {
            'total_predictions': total_predictions,
            'mbti_distribution': mbti_counts,
            'most_common': max(mbti_counts, key=mbti_counts.get) if mbti_counts else None
        }

    except Exception as e:
        logger.error(f"✗ Failed to get statistics: {e}")
        return None
    finally:
        db.close()


def get_recent_predictions(limit=10):
    """Get recent prediction records"""
    db = SessionLocal()
    try:
        predictions = db.query(PredictionData)\
            .order_by(PredictionData.timestamp.desc())\
            .limit(limit)\
            .all()

        results = []
        for p in predictions:
            results.append({
                'prediction_id': p.prediction_id,
                'timestamp': p.timestamp.isoformat(),
                'predicted_mbti': p.predicted_mbti,
                'dimension_scores': p.dimension_scores
            })

        return results

    except Exception as e:
        logger.error(f"✗ Failed to get prediction records: {e}")
        return []
    finally:
        db.close()


# ===================== Data Export Functions =====================

def export_training_data(output_file='training_data.json'):
    """Export all data for model training"""
    db = SessionLocal()
    try:
        predictions = db.query(PredictionData).all()

        training_data = []
        for p in predictions:
            training_data.append({
                'prediction_id': p.prediction_id,
                'timestamp': p.timestamp.isoformat(),
                'predicted_mbti': p.predicted_mbti,
                'dimension_scores': p.dimension_scores,
                'text_responses': p.text_responses,
                'emotion_data': p.emotion_data,
                'language': p.language
            })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Exported {len(training_data)} records to {output_file}")
        return True

    except Exception as e:
        logger.error(f"✗ Failed to export data: {e}")
        return False
    finally:
        db.close()
