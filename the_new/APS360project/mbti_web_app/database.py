"""
PostgreSQL 数据库管理模块
"""

import os
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import JSON
import logging

logger = logging.getLogger(__name__)

# 数据库配置
DATABASE_URL = os.getenv(
    'DATABASE_URL',
    'postgresql://postgres:password@localhost:5432/mbti_predictions'
)

# 创建 SQLAlchemy engine 和 session
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ===================== 数据模型定义 =====================

class PredictionData(Base):
    """预测数据表"""
    __tablename__ = 'prediction_data'

    prediction_id = Column(String(36), primary_key=True)  # UUID
    session_id = Column(String(36), index=True)  # 关联到 question session
    timestamp = Column(DateTime, default=datetime.utcnow)

    # 预测结果
    predicted_mbti = Column(String(4), index=True)

    # 各维度得分 (JSON)
    dimension_scores = Column(JSON)  # {"I": 0.6, "E": 0.4, "S": 0.7, ...}

    # 文本输入
    text_responses = Column(JSON)  # [{"question_id": 1, "response": "..."}]

    # 情绪数据 (JSON)
    emotion_data = Column(JSON)  # [{"timestamp": "...", "emotion": "happy", "confidence": 0.8}]

    # 元数据
    language = Column(String(10), default='en')
    question_version = Column(String(20))
    model_version = Column(String(20))

    # 关系
    question_usages = relationship("QuestionUsage", back_populates="prediction")


class QuestionUsage(Base):
    """问题使用记录表"""
    __tablename__ = 'question_usage'

    usage_id = Column(String(36), primary_key=True)
    prediction_id = Column(String(36), ForeignKey('prediction_data.prediction_id'))
    question_id = Column(Integer, index=True)
    question_order = Column(Integer)  # 第几个问题

    # 用户回答
    user_response_text = Column(Text)
    response_length = Column(Integer)

    # 关系
    prediction = relationship("PredictionData", back_populates="question_usages")


class UserSession(Base):
    """用户会话表"""
    __tablename__ = 'user_sessions'

    session_id = Column(String(36), primary_key=True)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)

    # 会话信息
    user_agent = Column(String(200))
    ip_address = Column(String(45))  # IPv6 support

    # 完成状态
    completed = Column(Boolean, default=False)


# ===================== 数据库初始化 =====================

def init_database():
    """初始化数据库，创建所有表"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ 数据库表创建成功")
        return True
    except Exception as e:
        logger.error(f"❌ 数据库初始化失败: {e}")
        return False


# ===================== 数据库操作函数 =====================

def get_db():
    """获取数据库会话"""
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
    保存预测结果到数据库

    参数:
        prediction_id: 预测ID (UUID)
        session_id: 会话ID
        predicted_mbti: 预测的MBTI类型
        dimension_scores: 各维度得分 {"I": 0.6, "E": 0.4, ...}
        text_responses: 文本回答 [{"question_id": 1, "response": "..."}]
        emotion_data: 情绪数据 (可选)
        language: 语言
        question_version: 问题版本
        model_version: 模型版本
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

        logger.info(f"✅ 保存预测结果: {prediction_id} -> {predicted_mbti}")
        return True

    except Exception as e:
        db.rollback()
        logger.error(f"❌ 保存预测失败: {e}")
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
    保存问题使用记录

    参数:
        usage_id: 使用记录ID (UUID)
        prediction_id: 关联的预测ID
        question_id: 问题ID
        question_order: 问题顺序（第几个）
        user_response_text: 用户回答文本
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

        logger.info(f"✅ 保存问题使用: Q{question_id} for {prediction_id}")
        return True

    except Exception as e:
        db.rollback()
        logger.error(f"❌ 保存问题使用失败: {e}")
        return False
    finally:
        db.close()


def create_user_session(
    session_id: str,
    user_agent: str = None,
    ip_address: str = None
):
    """创建用户会话记录"""
    db = SessionLocal()
    try:
        session = UserSession(
            session_id=session_id,
            user_agent=user_agent,
            ip_address=ip_address
        )

        db.add(session)
        db.commit()

        logger.info(f"✅ 创建会话: {session_id}")
        return True

    except Exception as e:
        db.rollback()
        logger.error(f"❌ 创建会话失败: {e}")
        return False
    finally:
        db.close()


def complete_user_session(session_id: str):
    """标记会话为已完成"""
    db = SessionLocal()
    try:
        session = db.query(UserSession).filter_by(session_id=session_id).first()
        if session:
            session.completed = True
            session.end_time = datetime.utcnow()
            db.commit()
            logger.info(f"✅ 会话完成: {session_id}")
            return True
        return False

    except Exception as e:
        db.rollback()
        logger.error(f"❌ 更新会话失败: {e}")
        return False
    finally:
        db.close()


# ===================== 统计查询函数 =====================

def get_prediction_statistics():
    """获取预测统计信息"""
    db = SessionLocal()
    try:
        total_predictions = db.query(PredictionData).count()

        # 各MBTI类型数量
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
        logger.error(f"❌ 获取统计失败: {e}")
        return None
    finally:
        db.close()


def get_recent_predictions(limit=10):
    """获取最近的预测记录"""
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
        logger.error(f"❌ 获取预测记录失败: {e}")
        return []
    finally:
        db.close()


# ===================== 导出数据函数 =====================

def export_training_data(output_file='training_data.json'):
    """导出所有数据用于模型训练"""
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

        logger.info(f"✅ 导出 {len(training_data)} 条数据到 {output_file}")
        return True

    except Exception as e:
        logger.error(f"❌ 导出数据失败: {e}")
        return False
    finally:
        db.close()
