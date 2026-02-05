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
import uuid
import sys
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory
from torchvision import transforms, models
from datetime import datetime

# 添加models目录到Python路径
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

# 导入问题管理模块
from questions import QuestionBank

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置模型文件路径
# 设置模型文件路径
BASE_DIR = r"E:\APS360_project\MBTI-Prediction-\the_new\APS360project\mbti_web_app\models"

# 情感识别模型 (PyTorch)
EMOTION_MODEL_PATH = os.path.join(BASE_DIR, "emotion", "improved_emotion_model.pth")

# 文本 MBTI 模型 (Scikit-learn)
TEXT_MODEL_DIR = os.path.join(BASE_DIR, "text", "ml")
TEXT_MODEL_PATH = os.path.join(TEXT_MODEL_DIR, "model.pkl")
TEXT_VECTORIZER_PATH = os.path.join(TEXT_MODEL_DIR, "vectorizer.pkl")
TEXT_LABEL_ENCODER_PATH = os.path.join(TEXT_MODEL_DIR, "label_encoder.pkl")

# 定义情绪类别
EMOTION_CATEGORIES = ['Anger', 'Confusion', 'Contempt', 'Disgust', 
                      'Happiness', 'Neutral', 'Sadness', 'Surprise']

# 创建 Flask 应用
app = Flask(__name__)

# 为存储结果创建目录
RESULTS_DIR = os.path.join(os.path.expanduser("~"), "MBTI_Results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ================ 问题库管理 ================

# 问题数据文件路径
QUESTIONS_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# 全局问题库变量（在初始化时加载）
question_bank_en = None
question_bank_zh = None

# ================ MBTI描述管理 ================

# MBTI描述数据（在初始化时加载）
mbti_descriptions_en = None
mbti_descriptions_zh = None

# ================ MediaPipe 人脸检测 ================

# 初始化 MediaPipe 人脸检测模块
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 创建全局 MediaPipe 人脸检测器
face_detector = None

def initialize_face_detector():
    """初始化 OpenCV DNN 人脸检测器（替代 MediaPipe）"""
    global face_detector

    # 模型文件路径
    model_dir = os.path.join(os.path.dirname(__file__), 'models', 'face_detection')
    prototxt_path = os.path.join(model_dir, 'deploy.prototxt')
    model_path = os.path.join(model_dir, 'res10_300x300_ssd_iter_140000.caffemodel')

    # 检查模型文件是否存在
    if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"人脸检测模型文件不存在。请运行: python download_face_models.py\n"
            f"缺少文件:\n"
            f"  - {prototxt_path}\n"
            f"  - {model_path}"
        )

    # 加载 OpenCV DNN 模型
    face_detector = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    logger.info("✅ 成功初始化 OpenCV DNN 人脸检测器（替代 MediaPipe）")

def detect_and_crop_face(image, padding=0.2):
    """
    使用 OpenCV DNN 检测并裁剪图像中的人脸

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

    h, w = image.shape[:2]

    # 预处理图像用于 DNN 模型
    # 创建 blob：resize to 300x300, subtract mean values
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    # 设置输入并进行前向传播
    face_detector.setInput(blob)
    detections = face_detector.forward()

    # 找到置信度最高的人脸
    best_confidence = 0
    best_box = None

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5 and confidence > best_confidence:
            best_confidence = confidence
            # 获取边界框坐标
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            best_box = box.astype("int")

    # 如果没有检测到人脸
    if best_box is None:
        return image, False

    # 解包坐标
    x1, y1, x2, y2 = best_box

    # 添加填充
    box_width = x2 - x1
    box_height = y2 - y1
    x1 = max(0, int(x1 - box_width * padding))
    y1 = max(0, int(y1 - box_height * padding))
    x2 = min(w, int(x2 + box_width * padding))
    y2 = min(h, int(y2 + box_height * padding))

    # 裁剪人脸区域
    face_crop = image[y1:y2, x1:x2]

    # 确保裁剪的人脸不为空
    if face_crop.size == 0:
        return image, False

    return face_crop, True

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

# ================ 问题管理 API ================

@app.route('/api/questions', methods=['GET'])
def get_questions():
    """
    获取随机问题用于测试会话

    查询参数:
        language (str): 'en' 或 'zh'，默认 'en'
        count (int): 问题数量，默认 3
        balanced (bool): 是否平衡维度，默认 true

    返回:
        JSON: {
            "questions": [...],
            "session_id": "...",
            "version": "1.0",
            "language": "..."
        }
    """
    try:
        # 获取参数
        language = request.args.get('language', 'en')
        count = int(request.args.get('count', 3))
        balanced = request.args.get('balanced', 'true').lower() == 'true'

        # 选择对应语言的问题库
        bank = question_bank_zh if language == 'zh' else question_bank_en

        if bank is None:
            logger.error(f"问题库未初始化 (language={language})")
            return jsonify({
                'error': 'Question bank not initialized',
                'language': language
            }), 500

        # 获取随机问题
        questions = bank.get_random_questions(count=count, balanced=balanced)

        # 生成会话ID用于追踪
        session_id = str(uuid.uuid4())

        logger.info(f"为会话 {session_id} 生成了 {len(questions)} 个问题 (语言={language}, 平衡={balanced})")

        return jsonify({
            'questions': [q.to_dict() for q in questions],
            'session_id': session_id,
            'version': '1.0',
            'language': language,
            'count': len(questions)
        })

    except Exception as e:
        logger.error(f"获取问题时出错: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/questions/<int:question_id>', methods=['GET'])
def get_question_by_id(question_id):
    """
    根据ID获取特定问题

    路径参数:
        question_id (int): 问题ID

    查询参数:
        language (str): 'en' 或 'zh'，默认 'en'

    返回:
        JSON: 问题详情
    """
    try:
        language = request.args.get('language', 'en')
        bank = question_bank_zh if language == 'zh' else question_bank_en

        if bank is None:
            return jsonify({'error': 'Question bank not initialized'}), 500

        question = bank.get_question(question_id)

        if question:
            logger.info(f"获取问题 ID={question_id} (语言={language})")
            return jsonify(question.to_dict())
        else:
            return jsonify({'error': 'Question not found'}), 404

    except Exception as e:
        logger.error(f"获取问题 {question_id} 时出错: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/questions/usage', methods=['POST'])
def record_question_usage():
    """
    记录用户回答了哪些问题

    请求体:
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

    返回:
        JSON: {"status": "success", "recorded": count}
    """
    try:
        data = request.json

        if not data or 'questions' not in data:
            return jsonify({'error': 'Missing questions data'}), 400

        prediction_id = data.get('prediction_id', 'unknown')
        session_id = data.get('session_id', 'unknown')
        questions = data.get('questions', [])

        # 记录日志（在生产环境中应保存到数据库）
        logger.info(f"记录问题使用情况: prediction_id={prediction_id}, session_id={session_id}, 问题数={len(questions)}")

        for q in questions:
            logger.debug(f"  问题 {q.get('question_id')}: 顺序={q.get('question_order')}, 回答长度={len(q.get('user_response', ''))}")

        # TODO: 在集成数据库后，保存到 question_usage 表
        # 格式:
        # INSERT INTO question_usage (usage_id, prediction_id, question_id, question_order, user_response_text, response_length)
        # VALUES (uuid, prediction_id, question_id, order, response, len(response))

        return jsonify({
            'status': 'success',
            'recorded': len(questions),
            'note': 'Currently logging only (database integration pending)'
        })

    except Exception as e:
        logger.error(f"记录问题使用时出错: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/questions/stats', methods=['GET'])
def get_question_stats():
    """
    获取问题库统计信息

    查询参数:
        language (str): 'en' 或 'zh'，默认 'en'

    返回:
        JSON: 问题库统计信息
    """
    try:
        language = request.args.get('language', 'en')
        bank = question_bank_zh if language == 'zh' else question_bank_en

        if bank is None:
            return jsonify({'error': 'Question bank not initialized'}), 500

        # 统计每个维度的问题数量
        stats = {
            'total_questions': len(bank.questions),
            'language': language,
            'by_dimension': {}
        }

        for dimension in ['I/E', 'S/N', 'T/F', 'J/P']:
            dim_questions = bank.get_questions_by_dimension(dimension, language, active_only=True)
            stats['by_dimension'][dimension] = len(dim_questions)

        logger.info(f"获取问题统计 (语言={language}): 总数={stats['total_questions']}")

        return jsonify(stats)

    except Exception as e:
        logger.error(f"获取问题统计时出错: {e}")
        return jsonify({'error': str(e)}), 500


# ================ MBTI Description API Endpoints ================

@app.route('/api/mbti/descriptions', methods=['GET'])
def get_mbti_descriptions():
    """
    获取所有MBTI类型描述

    查询参数:
        language (str): 'en' 或 'zh'，默认 'en'

    返回:
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

        # 选择对应语言的描述
        descriptions = mbti_descriptions_zh if language == 'zh' else mbti_descriptions_en

        if descriptions is None or not descriptions.get('types'):
            logger.error(f"MBTI描述未初始化 (language={language})")
            return jsonify({
                'error': 'MBTI descriptions not initialized',
                'language': language
            }), 500

        logger.info(f"获取MBTI描述 (语言={language}, 类型数={len(descriptions.get('types', {}))})")

        return jsonify(descriptions)

    except Exception as e:
        logger.error(f"获取MBTI描述时出错: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/mbti/descriptions/<mbti_type>', methods=['GET'])
def get_mbti_description_by_type(mbti_type):
    """
    根据MBTI类型获取描述

    路径参数:
        mbti_type (str): MBTI类型，如 'INTJ', 'ENFP' 等

    查询参数:
        language (str): 'en' 或 'zh'，默认 'en'

    返回:
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

        # 验证MBTI类型格式
        if len(mbti_type) != 4 or not all(c in 'IESTFNJP' for c in mbti_type):
            return jsonify({'error': 'Invalid MBTI type format'}), 400

        # 选择对应语言的描述
        descriptions = mbti_descriptions_zh if language == 'zh' else mbti_descriptions_en

        if descriptions is None:
            return jsonify({'error': 'MBTI descriptions not initialized'}), 500

        # 获取特定类型的描述
        type_desc = descriptions.get('types', {}).get(mbti_type)

        if type_desc:
            logger.info(f"获取MBTI描述: {mbti_type} (语言={language})")
            result = {
                'type': mbti_type,
                **type_desc
            }
            return jsonify(result)
        else:
            return jsonify({'error': f'MBTI type {mbti_type} not found'}), 404

    except Exception as e:
        logger.error(f"获取MBTI描述 {mbti_type} 时出错: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/test')
def test_page():
    """测试页面，用于验证服务器运行正常"""
    return "MBTI 评估系统服务器运行正常！"

# ================ 应用启动 ================

def load_question_banks():
    """加载问题库"""
    global question_bank_en, question_bank_zh

    try:
        # 英文问题库
        en_file = os.path.join(QUESTIONS_DATA_DIR, 'questions_en.json')
        if os.path.exists(en_file):
            question_bank_en = QuestionBank(en_file)
            logger.info(f"✅ 成功加载英文问题库: {len(question_bank_en.questions)} 个问题")
        else:
            logger.warning(f"⚠️  英文问题文件不存在: {en_file}")
            # 创建空问题库作为备用
            question_bank_en = QuestionBank()

        # 中文问题库
        zh_file = os.path.join(QUESTIONS_DATA_DIR, 'questions_zh.json')
        if os.path.exists(zh_file):
            question_bank_zh = QuestionBank(zh_file)
            logger.info(f"✅ 成功加载中文问题库: {len(question_bank_zh.questions)} 个问题")
        else:
            logger.warning(f"⚠️  中文问题文件不存在: {zh_file}")
            # 创建空问题库作为备用
            question_bank_zh = QuestionBank()

        return True

    except Exception as e:
        logger.error(f"❌ 加载问题库时出错: {e}")
        # 创建空问题库以防止崩溃
        question_bank_en = QuestionBank()
        question_bank_zh = QuestionBank()
        return False


def load_mbti_descriptions():
    """加载MBTI类型描述"""
    global mbti_descriptions_en, mbti_descriptions_zh

    try:
        # 英文描述
        en_file = os.path.join(QUESTIONS_DATA_DIR, 'mbti_descriptions_en.json')
        if os.path.exists(en_file):
            with open(en_file, 'r', encoding='utf-8') as f:
                mbti_descriptions_en = json.load(f)
            logger.info(f"✅ 成功加载英文MBTI描述: {len(mbti_descriptions_en.get('types', {}))} 个类型")
        else:
            logger.warning(f"⚠️  英文MBTI描述文件不存在: {en_file}")
            mbti_descriptions_en = {"types": {}}

        # 中文描述
        zh_file = os.path.join(QUESTIONS_DATA_DIR, 'mbti_descriptions_zh.json')
        if os.path.exists(zh_file):
            with open(zh_file, 'r', encoding='utf-8') as f:
                mbti_descriptions_zh = json.load(f)
            logger.info(f"✅ 成功加载中文MBTI描述: {len(mbti_descriptions_zh.get('types', {}))} 个类型")
        else:
            logger.warning(f"⚠️  中文MBTI描述文件不存在: {zh_file}")
            mbti_descriptions_zh = {"types": {}}

        return True

    except Exception as e:
        logger.error(f"❌ 加载MBTI描述时出错: {e}")
        mbti_descriptions_en = {"types": {}}
        mbti_descriptions_zh = {"types": {}}
        return False


def initialize_app():
    """初始化应用，加载所有模型"""
    logger.info("正在初始化 MBTI 评估系统...")

    # 初始化人脸检测器（允许失败）
    face_detector_loaded = False
    try:
        initialize_face_detector()
        face_detector_loaded = True
    except Exception as e:
        logger.warning(f"⚠️  人脸检测器初始化失败: {e}")
        logger.info("  ℹ️  情绪识别功能将不可用，但问题管理API仍可正常工作")

    # 加载情绪模型（允许失败）
    emotion_model_loaded = False
    try:
        emotion_model_loaded = load_emotion_model()
    except Exception as e:
        logger.warning(f"⚠️  情绪模型加载失败: {e}")

    # 加载文本模型（允许失败）
    text_model_loaded = False
    try:
        text_model_loaded = load_text_model()
    except Exception as e:
        logger.warning(f"⚠️  文本模型加载失败: {e}")

    # 加载问题库（这个是必须的）
    questions_loaded = load_question_banks()

    # 加载MBTI描述
    descriptions_loaded = load_mbti_descriptions()

    # 输出状态总结
    logger.info("\n" + "="*60)
    logger.info("组件加载状态:")
    logger.info(f"  {'✅' if face_detector_loaded else '❌'} 人脸检测器: {'已加载' if face_detector_loaded else '未加载'}")
    logger.info(f"  {'✅' if emotion_model_loaded else '❌'} 情绪识别模型: {'已加载' if emotion_model_loaded else '未加载'}")
    logger.info(f"  {'✅' if text_model_loaded else '❌'} 文本MBTI模型: {'已加载' if text_model_loaded else '未加载'}")
    logger.info(f"  {'✅' if questions_loaded else '❌'} 问题库: {'已加载' if questions_loaded else '未加载'}")
    logger.info(f"  {'✅' if descriptions_loaded else '❌'} MBTI描述: {'已加载' if descriptions_loaded else '未加载'}")
    logger.info("="*60)

    if questions_loaded:
        logger.info("✅ 问题管理API已就绪！")
        logger.info("   可用端点: GET /api/questions, /api/questions/<id>, /api/questions/stats")

    if descriptions_loaded:
        logger.info("✅ MBTI描述API已就绪！")
        logger.info("   可用端点: GET /api/mbti/descriptions, /api/mbti/descriptions/<type>")

    if emotion_model_loaded and text_model_loaded and questions_loaded:
        logger.info("✅ 所有模型和问题库加载成功！完整功能可用。")
    elif questions_loaded:
        logger.warning("⚠️  部分功能不可用，但问题管理API正常工作")
        logger.info("   💡 提示: 您仍然可以测试新的问题管理功能！")
    else:
        logger.error("❌ 问题库加载失败，系统无法启动")
        sys.exit(1)

if __name__ == '__main__':
    # 初始化应用
    initialize_app()
    
    # 启动服务器
    app.run(debug=True, host='0.0.0.0', port=5000)