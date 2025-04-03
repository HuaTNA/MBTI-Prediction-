import pandas as pd
import numpy as np
import re
import os
import joblib
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
import nltk

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import logging

# 忽略警告
warnings.filterwarnings("ignore")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mbti_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 设置随机种子以获得可重复的结果
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# 检查是否有GPU可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {device}")

# 定义输出目录
OUTPUT_DIR = "mbti_enhanced_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#################################################
# 自定义词汇表实现 (替代 torchtext)
#################################################

class Vocabulary:
    """简单的词汇表类，替代 torchtext.vocab"""
    
    def __init__(self):
        # 初始化特殊标记
        self.word2idx = {"<unk>": 0, "<pad>": 1}
        self.idx2word = {0: "<unk>", 1: "<pad>"}
        self.word_count = {}
        self.default_index = 0  # <unk> token index
        
    def build_vocab(self, texts, tokenizer, min_freq=2):
        """从文本构建词汇表"""
        # 计算词频
        for text in texts:
            for token in tokenizer(text):
                self.word_count[token] = self.word_count.get(token, 0) + 1
        
        # 添加符合最小频率的词
        idx = len(self.word2idx)
        for word, count in self.word_count.items():
            if count >= min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        
        logger.info(f"词汇表大小: {len(self.word2idx)}")
        return self
    
    def __getitem__(self, token):
        """获取词的索引，如果不存在则返回默认索引"""
        return self.word2idx.get(token, self.default_index)
    
    def __len__(self):
        """返回词汇表大小"""
        return len(self.word2idx)
    
    def set_default_index(self, idx):
        """设置默认索引"""
        self.default_index = idx

#################################################
# 数据处理和特征工程
#################################################

class TextProcessor:
    """文本预处理和特征提取类"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
    
    def basic_clean(self, text):
        """基本文本清理"""
        if not isinstance(text, str):
            return ""
        
        # 转为小写
        text = text.lower()
        
        # 移除URL
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # 移除HTML标签
        text = re.sub(r'<.*?>', '', text)
        
        # 移除特殊字符和数字
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def advanced_clean(self, text):
        """高级文本清理，包括停用词去除和词干提取"""
        # 基础清理
        text = self.basic_clean(text)
        
        # 分词
        words = word_tokenize(text)
        
        # 停用词去除
        words = [w for w in words if w not in self.stop_words]
        
        # 词形还原
        words = [self.lemmatizer.lemmatize(w) for w in words]
        
        return ' '.join(words)
    
    def tokenize(self, text):
        """将文本分词并返回分词列表"""
        text = self.basic_clean(text)
        return word_tokenize(text)
    
    def extract_features(self, text):
        """提取文本特征"""
        features = {}
        
        # 文本长度特征
        features['text_length'] = len(text)
        
        # 句子数量
        sentences = text.split('.')
        features['sentence_count'] = len(sentences)
        
        # 词汇特征
        words = text.split()
        if words:
            features['word_count'] = len(words)
            features['avg_word_length'] = sum(len(w) for w in words) / len(words)
        else:
            features['word_count'] = 0
            features['avg_word_length'] = 0
        
        # 情感分析
        try:
            blob = TextBlob(text)
            features['sentiment'] = blob.sentiment.polarity
            features['subjectivity'] = blob.sentiment.subjectivity
        except:
            features['sentiment'] = 0
            features['subjectivity'] = 0
        
        # 标点符号频率
        original_text = text if isinstance(text, str) else ""
        features['question_ratio'] = original_text.count('?') / (len(original_text) + 1) * 100
        features['exclamation_ratio'] = original_text.count('!') / (len(original_text) + 1) * 100
        
        # 大写字母比例（可能表明情绪强度）
        if original_text:
            upper_chars = sum(1 for c in original_text if c.isupper())
            features['uppercase_ratio'] = upper_chars / len(original_text) * 100
        else:
            features['uppercase_ratio'] = 0
        
        return features

def build_vocabulary(texts, tokenizer, min_freq=2):
    """构建词汇表"""
    vocab = Vocabulary()
    return vocab.build_vocab(texts, tokenizer, min_freq)

def load_and_process_data(file_path, processor, test_size=0.2, val_size=0.15):
    """加载和处理MBTI数据集，包括特征提取和维度分解"""
    logger.info(f"正在从 {file_path} 加载数据...")
    
    try:
        # 加载数据
        df = pd.read_csv(file_path)
        logger.info(f"成功加载数据: {len(df)} 行")
        
        # 确保列名正确
        if df.columns[0] != 'type' or df.columns[1] != 'posts':
            df.columns = ['type', 'posts'] + list(df.columns[2:])
        
        # 转换MBTI类型为大写
        df['type'] = df['type'].str.upper()
        
        # 过滤无效的MBTI类型
        valid_types = [
            'ISTJ', 'ISFJ', 'INFJ', 'INTJ', 
            'ISTP', 'ISFP', 'INFP', 'INTP', 
            'ESTP', 'ESFP', 'ENFP', 'ENTP', 
            'ESTJ', 'ESFJ', 'ENFJ', 'ENTJ'
        ]
        
        original_len = len(df)
        df = df[df['type'].isin(valid_types)]
        logger.info(f"过滤无效的MBTI类型后: {len(df)} 行 (移除了 {original_len - len(df)} 行)")
        
        # 显示类别分布
        type_counts = df['type'].value_counts()
        logger.info("\nMBTI类型分布:")
        for mbti_type, count in type_counts.items():
            logger.info(f"{mbti_type}: {count} ({count/len(df)*100:.2f}%)")
        
        # 创建基本清理文本
        logger.info("正在清理文本...")
        df['cleaned_posts'] = df['posts'].apply(processor.basic_clean)
        
        # 创建高级清理文本
        logger.info("正在进行高级文本处理...")
        df['advanced_posts'] = df['posts'].apply(processor.advanced_clean)
        
        # 提取文本特征
        logger.info("正在提取文本特征...")
        features = df['posts'].apply(processor.extract_features)
        features_df = pd.DataFrame(features.tolist())
        
        # 将特征合并到主数据框
        df = pd.concat([df, features_df], axis=1)
        
        # 分解MBTI维度
        logger.info("正在分解MBTI维度...")
        df['e_i'] = df['type'].str[0]  # 提取第一个字母（E/I）
        df['n_s'] = df['type'].str[1]  # 提取第二个字母（N/S）
        df['t_f'] = df['type'].str[2]  # 提取第三个字母（T/F）
        df['j_p'] = df['type'].str[3]  # 提取第四个字母（J/P）
        
        # 对完整MBTI类型进行编码
        label_encoder = LabelEncoder()
        df['encoded_type'] = label_encoder.fit_transform(df['type'])
        
        # 对各个维度进行编码
        dim_encoders = {}
        for dim in ['e_i', 'n_s', 't_f', 'j_p']:
            dim_encoder = LabelEncoder()
            df[f'encoded_{dim}'] = dim_encoder.fit_transform(df[dim])
            dim_encoders[dim] = dim_encoder
        
        # 可视化MBTI类型分布
        plt.figure(figsize=(12, 6))
        sns.countplot(x='type', data=df)
        plt.title('MBTI类型分布')
        plt.xlabel('MBTI类型')
        plt.ylabel('样本数量')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/mbti_distribution.png")
        plt.close()
        
        # 可视化维度分布
        plt.figure(figsize=(15, 5))
        for i, dim in enumerate(['e_i', 'n_s', 't_f', 'j_p']):
            plt.subplot(1, 4, i+1)
            sns.countplot(x=dim, data=df)
            plt.title(f'{dim} 维度分布')
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/dimensions_distribution.png")
        plt.close()
        
        # 构建词汇表
        logger.info("构建词汇表...")
        vocab = build_vocabulary(df['cleaned_posts'].tolist(), processor.tokenize)
        logger.info(f"词汇表大小: {len(vocab)}")
        
        # 分割数据集
        logger.info("正在分割数据集...")
        
        # 检查是否可以使用分层抽样
        min_count = type_counts.min()
        if min_count < 2:
            logger.warning("某些MBTI类型的样本少于2个，禁用分层抽样")
            X_train, X_test, y_train, y_test = train_test_split(
                df[['cleaned_posts', 'advanced_posts'] + list(features_df.columns)], 
                df[['encoded_type', 'type', 'encoded_e_i', 'encoded_n_s', 'encoded_t_f', 'encoded_j_p']],
                test_size=test_size,
                random_state=RANDOM_SEED
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, 
                y_train,
                test_size=val_size,
                random_state=RANDOM_SEED
            )
        else:
            logger.info("使用分层抽样保持各类型比例...")
            X_train, X_test, y_train, y_test = train_test_split(
                df[['cleaned_posts', 'advanced_posts'] + list(features_df.columns)], 
                df[['encoded_type', 'type', 'encoded_e_i', 'encoded_n_s', 'encoded_t_f', 'encoded_j_p']],
                test_size=test_size,
                random_state=RANDOM_SEED,
                stratify=df['type']
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, 
                y_train,
                test_size=val_size,
                random_state=RANDOM_SEED,
                stratify=y_train['type']
            )
        
        logger.info(f"训练集大小: {len(X_train)}")
        logger.info(f"验证集大小: {len(X_val)}")
        logger.info(f"测试集大小: {len(X_test)}")
        
        # 计算类权重
        class_counts = Counter(y_train['type'])
        # 增强版类权重，使用指数1.5增强稀有类型的权重
        class_weights = {mbti_type: (len(y_train) / (count * len(class_counts)))**1.5
                        for mbti_type, count in class_counts.items()}
        
        logger.info("类别权重:")
        for mbti_type, weight in class_weights.items():
            logger.info(f"{mbti_type}: {weight:.4f}")
        
        # 返回处理后的数据
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'label_encoder': label_encoder,
            'dim_encoders': dim_encoders,
            'class_weights': class_weights,
            'num_classes': len(label_encoder.classes_),
            'df': df,
            'vocab': vocab,
            'processor': processor
        }
        
    except Exception as e:
        logger.error(f"加载或处理数据时出错: {e}")
        raise

#################################################
# 数据集类
#################################################

class MBTIDataset(Dataset):
    """用于深度学习模型的MBTI数据集类"""
    def __init__(self, texts, labels, vocab, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 分词
        tokens = self.tokenizer(text)
        
        # 转换为索引
        indices = [self.vocab[token] for token in tokens]
        
        # 截断或填充
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        else:
            indices = indices + [self.vocab["<pad>"]] * (self.max_length - len(indices))
        
        # 转换为张量
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        # 创建掩码（实际长度为1，填充为0）
        mask = torch.zeros(self.max_length, dtype=torch.long)
        mask[:min(len(tokens), self.max_length)] = 1
        
        return {
            'input_ids': indices_tensor,
            'attention_mask': mask,
            'label': label_tensor
        }

class MBTIDimensionDataset(Dataset):
    """用于MBTI维度分类的数据集类"""
    def __init__(self, texts, dim_labels, vocab, tokenizer, max_length=512):
        self.texts = texts
        self.dim_labels = dim_labels  # 维度标签 (e_i, n_s, t_f, j_p)
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # 分词
        tokens = self.tokenizer(text)
        
        # 转换为索引
        indices = [self.vocab[token] for token in tokens]
        
        # 截断或填充
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        else:
            indices = indices + [self.vocab["<pad>"]] * (self.max_length - len(indices))
        
        # 转换为张量
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        
        # 创建掩码（实际长度为1，填充为0）
        mask = torch.zeros(self.max_length, dtype=torch.long)
        mask[:min(len(tokens), self.max_length)] = 1
        
        # 准备所有维度的标签
        result = {
            'input_ids': indices_tensor,
            'attention_mask': mask,
        }
        
        # 添加各维度标签
        for dim, labels in self.dim_labels.items():
            if dim in ['e_i', 'n_s', 't_f', 'j_p']:
                dim_key = f'{dim}_label'
                if dim_key in labels:
                    result[dim_key] = torch.tensor(labels[dim_key][idx], dtype=torch.long)
        
        return result

#################################################
# 模型定义
#################################################

class LSTMMBTIClassifier(nn.Module):
    """基于LSTM的MBTI分类器"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=2, dropout_rate=0.3):
        super(LSTMMBTIClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            bidirectional=True, 
            dropout=dropout_rate if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 因为是双向LSTM
    
    def forward(self, input_ids, attention_mask=None):
        # 将输入转换为嵌入向量
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        
        # 应用LSTM
        if attention_mask is not None:
            # 计算序列长度
            lengths = attention_mask.sum(dim=1).cpu()
            
            # 打包序列
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
            
            packed_output, (hidden, cell) = self.lstm(packed_embedded)
            
            # 解包序列
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            output, (hidden, cell) = self.lstm(embedded)
        
        # 获取最后一个时间步的隐藏状态（考虑双向）
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [batch_size, hidden_dim*2]
        
        # 应用dropout
        hidden = self.dropout(hidden)
        
        # 最终分类
        return self.fc(hidden)

class CNNMBTIClassifier(nn.Module):
    """基于CNN的MBTI分类器"""
    def __init__(self, vocab_size, embedding_dim, num_classes, filter_sizes=(3, 4, 5), num_filters=100, dropout_rate=0.3):
        super(CNNMBTIClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        
        # 多尺寸卷积
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, k) for k in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
    
    def forward(self, input_ids, attention_mask=None):
        # 将输入转换为嵌入向量
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        
        # 转换为卷积所需的格式
        embedded = embedded.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_len]
        
        # 应用卷积和最大池化
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        # 连接不同尺寸的特征
        cat = torch.cat(pooled, dim=1)
        
        # 应用dropout
        cat = self.dropout(cat)
        
        # 最终分类
        return self.fc(cat)

class GRUMBTIClassifier(nn.Module):
    """基于GRU的MBTI分类器"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=2, dropout_rate=0.3):
        super(GRUMBTIClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.gru = nn.GRU(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            bidirectional=True, 
            dropout=dropout_rate if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 因为是双向GRU
    
    def forward(self, input_ids, attention_mask=None):
        # 将输入转换为嵌入向量
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        
        # 应用GRU
        if attention_mask is not None:
            # 计算序列长度
            lengths = attention_mask.sum(dim=1).cpu()
            
            # 打包序列
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
            
            packed_output, hidden = self.gru(packed_embedded)
            
            # 解包序列
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            output, hidden = self.gru(embedded)
        
        # 获取最后一个时间步的隐藏状态（考虑双向）
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [batch_size, hidden_dim*2]
        
        # 应用dropout
        hidden = self.dropout(hidden)
        
        # 最终分类
        return self.fc(hidden)

class HybridMBTIClassifier(nn.Module):
    """混合CNN和RNN的MBTI分类器"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, filter_sizes=(3, 4, 5), num_filters=100, dropout_rate=0.3):
        super(HybridMBTIClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        
        # CNN部分
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, k) for k in filter_sizes
        ])
        
        # RNN部分
        self.lstm = nn.LSTM(
            len(filter_sizes) * num_filters, 
            hidden_dim, 
            bidirectional=True, 
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 因为是双向LSTM
    
    def forward(self, input_ids, attention_mask=None):
        # 将输入转换为嵌入向量
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        
        # 转换为卷积所需的格式
        embedded = embedded.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_len]
        
        # 应用卷积和最大池化
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        # 连接不同尺寸的特征
        cat = torch.cat(pooled, dim=1)  # [batch_size, filters * len(filter_sizes)]
        
        # 扩展为序列以输入LSTM
        cat = cat.unsqueeze(1)  # [batch_size, 1, filters * len(filter_sizes)]
        
        # 应用LSTM
        output, (hidden, cell) = self.lstm(cat)
        
        # 获取最后一个时间步的隐藏状态（考虑双向）
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [batch_size, hidden_dim*2]
        
        # 应用dropout
        hidden = self.dropout(hidden)
        
        # 最终分类
        return self.fc(hidden)

class DimensionClassifier(nn.Module):
    """MBTI维度分类器（二分类）"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, model_type='lstm', dropout_rate=0.3):
        super(DimensionClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.model_type = model_type
        
        # 根据选择的模型类型初始化基础模型
        if model_type == 'lstm':
            self.base_model = nn.LSTM(
                embedding_dim, 
                hidden_dim, 
                num_layers=2, 
                bidirectional=True, 
                dropout=dropout_rate,
                batch_first=True
            )
            self.feature_dim = hidden_dim * 2  # 双向LSTM
        elif model_type == 'gru':
            self.base_model = nn.GRU(
                embedding_dim, 
                hidden_dim, 
                num_layers=2, 
                bidirectional=True, 
                dropout=dropout_rate,
                batch_first=True
            )
            self.feature_dim = hidden_dim * 2  # 双向GRU
        elif model_type == 'cnn':
            self.filter_sizes = (3, 4, 5)
            self.num_filters = 100
            self.convs = nn.ModuleList([
                nn.Conv1d(embedding_dim, self.num_filters, k) for k in self.filter_sizes
            ])
            self.feature_dim = len(self.filter_sizes) * self.num_filters
        elif model_type == 'hybrid':
            self.filter_sizes = (3, 4, 5)
            self.num_filters = 100
            self.convs = nn.ModuleList([
                nn.Conv1d(embedding_dim, self.num_filters, k) for k in self.filter_sizes
            ])
            self.lstm = nn.LSTM(
                len(self.filter_sizes) * self.num_filters, 
                hidden_dim, 
                bidirectional=True, 
                batch_first=True
            )
            self.feature_dim = hidden_dim * 2  # 双向LSTM
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # 为每个维度创建二分类器
        self.e_i_classifier = nn.Linear(self.feature_dim, 2)
        self.n_s_classifier = nn.Linear(self.feature_dim, 2)
        self.t_f_classifier = nn.Linear(self.feature_dim, 2)
        self.j_p_classifier = nn.Linear(self.feature_dim, 2)
    
    def _extract_features(self, input_ids, attention_mask=None):
        """提取特征"""
        # 将输入转换为嵌入向量
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        
        if self.model_type in ['lstm', 'gru']:
            # RNN模型
            if attention_mask is not None:
                # 计算序列长度
                lengths = attention_mask.sum(dim=1).cpu()
                
                # 打包序列
                packed_embedded = nn.utils.rnn.pack_padded_sequence(
                    embedded, lengths, batch_first=True, enforce_sorted=False
                )
                
                if self.model_type == 'lstm':
                    packed_output, (hidden, cell) = self.base_model(packed_embedded)
                else:  # gru
                    packed_output, hidden = self.base_model(packed_embedded)
                
                # 获取最后一个时间步的隐藏状态（考虑双向）
                features = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [batch_size, hidden_dim*2]
            else:
                if self.model_type == 'lstm':
                    output, (hidden, cell) = self.base_model(embedded)
                else:  # gru
                    output, hidden = self.base_model(embedded)
                
                # 获取最后一个时间步的隐藏状态（考虑双向）
                features = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [batch_size, hidden_dim*2]
        
        elif self.model_type == 'cnn':
            # CNN模型
            # 转换为卷积所需的格式
            embedded = embedded.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_len]
            
            # 应用卷积和最大池化
            conved = [F.relu(conv(embedded)) for conv in self.convs]
            pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
            
            # 连接不同尺寸的特征
            features = torch.cat(pooled, dim=1)  # [batch_size, filters * len(filter_sizes)]
        
        elif self.model_type == 'hybrid':
            # 混合CNN+LSTM模型
            # 转换为卷积所需的格式
            embedded = embedded.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_len]
            
            # 应用卷积和最大池化
            conved = [F.relu(conv(embedded)) for conv in self.convs]
            pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
            
            # 连接不同尺寸的特征
            cat = torch.cat(pooled, dim=1)  # [batch_size, filters * len(filter_sizes)]
            
            # 扩展为序列以输入LSTM
            cat = cat.unsqueeze(1)  # [batch_size, 1, filters * len(filter_sizes)]
            
            # 应用LSTM
            output, (hidden, cell) = self.lstm(cat)
            
            # 获取最后一个时间步的隐藏状态（考虑双向）
            features = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [batch_size, hidden_dim*2]
        
        # 应用dropout
        features = self.dropout(features)
        
        return features
    
    def forward(self, input_ids, attention_mask=None, dimension=None):
        # 提取特征
        features = self._extract_features(input_ids, attention_mask)
        
        # 如果指定了维度，只返回该维度的分类结果
        if dimension == 'e_i':
            return self.e_i_classifier(features)
        elif dimension == 'n_s':
            return self.n_s_classifier(features)
        elif dimension == 't_f':
            return self.t_f_classifier(features)
        elif dimension == 'j_p':
            return self.j_p_classifier(features)
        
        # 否则返回所有维度的分类结果
        return {
            'e_i': self.e_i_classifier(features),
            'n_s': self.n_s_classifier(features),
            't_f': self.t_f_classifier(features),
            'j_p': self.j_p_classifier(features)
        }

#################################################
# 训练函数
#################################################

def train_model(model, train_loader, val_loader, optimizer, criterion, 
                num_epochs, device, model_name):
    """训练深度学习模型的通用函数"""
    logger.info(f"开始训练 {model_name} 模型...")
    
    best_val_accuracy = 0.0
    training_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_loop = tqdm(train_loader, desc=f'第 {epoch+1}/{num_epochs} 轮 [训练]')
        for batch in train_loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(input_ids, attention_mask)
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新权重
            optimizer.step()
            
            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # 更新训练损失
            train_loss += loss.item() * input_ids.size(0)
            
            # 更新进度条
            train_loop.set_postfix(loss=train_loss/train_total, accuracy=train_correct/train_total)
        
        train_loss /= train_total
        train_accuracy = train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f'第 {epoch+1}/{num_epochs} 轮 [验证]')
            for batch in val_loop:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                # 前向传播
                outputs = model(input_ids, attention_mask)
                
                # 计算损失
                loss = criterion(outputs, labels)
                
                # 计算准确率
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # 更新验证损失
                val_loss += loss.item() * input_ids.size(0)
                
                # 更新进度条
                val_loop.set_postfix(loss=val_loss/val_total, accuracy=val_correct/val_total)
        
        val_loss /= val_total
        val_accuracy = val_correct / val_total
        
        # 保存训练历史
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_accuracy)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_accuracy)
        
        logger.info(f'第 {epoch+1}/{num_epochs} 轮:')
        logger.info(f'  训练损失: {train_loss:.4f}, 训练准确率: {train_accuracy:.4f}')
        logger.info(f'  验证损失: {val_loss:.4f}, 验证准确率: {val_accuracy:.4f}')
        
        # 保存最佳模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), f'{OUTPUT_DIR}/{model_name}_best.pt')
            logger.info(f'  保存新的最佳模型，验证准确率: {val_accuracy:.4f}')
    
    # 绘制训练历史
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(training_history['train_loss'], label='训练损失')
    plt.plot(training_history['val_loss'], label='验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.title(f'{model_name} 损失曲线')
    
    plt.subplot(1, 2, 2)
    plt.plot(training_history['train_acc'], label='训练准确率')
    plt.plot(training_history['val_acc'], label='验证准确率')
    plt.xlabel('轮次')
    plt.ylabel('准确率')
    plt.legend()
    plt.title(f'{model_name} 准确率曲线')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{model_name}_training_history.png')
    plt.close()
    
    # 加载最佳模型
    model.load_state_dict(torch.load(f'{OUTPUT_DIR}/{model_name}_best.pt'))
    
    return model, best_val_accuracy

def train_dimension_model(model, train_loader, val_loader, optimizer, criterion, 
                        num_epochs, device, model_name):
    """训练MBTI维度分类模型"""
    logger.info(f"开始训练 {model_name} 维度模型...")
    
    best_val_accuracy = 0.0
    training_history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
        'e_i_acc': [], 'n_s_acc': [], 't_f_acc': [], 'j_p_acc': []
    }
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_total = 0
        dimension_correct = {'e_i': 0, 'n_s': 0, 't_f': 0, 'j_p': 0}
        
        train_loop = tqdm(train_loader, desc=f'第 {epoch+1}/{num_epochs} 轮 [训练]')
        for batch in train_loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 获取所有维度的标签
            dimension_labels = {
                'e_i': batch['e_i_label'].to(device) if 'e_i_label' in batch else None,
                'n_s': batch['n_s_label'].to(device) if 'n_s_label' in batch else None,
                't_f': batch['t_f_label'].to(device) if 't_f_label' in batch else None,
                'j_p': batch['j_p_label'].to(device) if 'j_p_label' in batch else None
            }
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(input_ids, attention_mask)
            
            # 计算损失（每个维度的损失总和）
            loss = 0
            for dim, labels in dimension_labels.items():
                if labels is not None:
                    dim_loss = criterion(outputs[dim], labels)
                    loss += dim_loss
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新权重
            optimizer.step()
            
            # 计算准确率
            batch_size = input_ids.size(0)
            train_total += batch_size
            
            for dim, labels in dimension_labels.items():
                if labels is not None:
                    _, predicted = torch.max(outputs[dim], 1)
                    dimension_correct[dim] += (predicted == labels).sum().item()
            
            # 更新训练损失
            train_loss += loss.item() * batch_size
            
            # 计算平均准确率
            dimension_accuracies = {dim: dimension_correct[dim] / train_total 
                                  for dim in dimension_correct}
            avg_accuracy = sum(dimension_accuracies.values()) / len(dimension_accuracies)
            
            # 更新进度条
            train_loop.set_postfix(loss=train_loss/train_total, accuracy=avg_accuracy)
        
        train_loss /= train_total
        dimension_accuracies = {dim: dimension_correct[dim] / train_total 
                              for dim in dimension_correct}
        train_accuracy = sum(dimension_accuracies.values()) / len(dimension_accuracies)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_total = 0
        val_dimension_correct = {'e_i': 0, 'n_s': 0, 't_f': 0, 'j_p': 0}
        
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f'第 {epoch+1}/{num_epochs} 轮 [验证]')
            for batch in val_loop:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # 获取所有维度的标签
                dimension_labels = {
                    'e_i': batch['e_i_label'].to(device) if 'e_i_label' in batch else None,
                    'n_s': batch['n_s_label'].to(device) if 'n_s_label' in batch else None,
                    't_f': batch['t_f_label'].to(device) if 't_f_label' in batch else None,
                    'j_p': batch['j_p_label'].to(device) if 'j_p_label' in batch else None
                }
                
                # 前向传播
                outputs = model(input_ids, attention_mask)
                
                # 计算损失（每个维度的损失总和）
                loss = 0
                for dim, labels in dimension_labels.items():
                    if labels is not None:
                        dim_loss = criterion(outputs[dim], labels)
                        loss += dim_loss
                
                # 计算准确率
                batch_size = input_ids.size(0)
                val_total += batch_size
                
                for dim, labels in dimension_labels.items():
                    if labels is not None:
                        _, predicted = torch.max(outputs[dim], 1)
                        val_dimension_correct[dim] += (predicted == labels).sum().item()
                
                # 更新验证损失
                val_loss += loss.item() * batch_size
                
                # 计算平均准确率
                val_dimension_accuracies = {dim: val_dimension_correct[dim] / val_total 
                                          for dim in val_dimension_correct}
                val_avg_accuracy = sum(val_dimension_accuracies.values()) / len(val_dimension_accuracies)
                
                # 更新进度条
                val_loop.set_postfix(loss=val_loss/val_total, accuracy=val_avg_accuracy)
        
        val_loss /= val_total
        val_dimension_accuracies = {dim: val_dimension_correct[dim] / val_total 
                                   for dim in val_dimension_correct}
        val_accuracy = sum(val_dimension_accuracies.values()) / len(val_dimension_accuracies)
        
        # 保存训练历史
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_accuracy)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_accuracy)
        
        for dim in val_dimension_accuracies:
            training_history[f'{dim}_acc'].append(val_dimension_accuracies[dim])
        
        logger.info(f'第 {epoch+1}/{num_epochs} 轮:')
        logger.info(f'  训练损失: {train_loss:.4f}, 训练准确率: {train_accuracy:.4f}')
        logger.info(f'  验证损失: {val_loss:.4f}, 验证准确率: {val_accuracy:.4f}')
        logger.info(f'  维度准确率: {", ".join([f"{dim}: {acc:.4f}" for dim, acc in val_dimension_accuracies.items()])}')
        
        # 保存最佳模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), f'{OUTPUT_DIR}/{model_name}_best.pt')
            logger.info(f'  保存新的最佳模型，验证准确率: {val_accuracy:.4f}')
    
    # 绘制训练历史
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(training_history['train_loss'], label='训练损失')
    plt.plot(training_history['val_loss'], label='验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.title(f'{model_name} 损失曲线')
    
    plt.subplot(2, 2, 2)
    plt.plot(training_history['train_acc'], label='训练准确率')
    plt.plot(training_history['val_acc'], label='验证准确率')
    plt.xlabel('轮次')
    plt.ylabel('准确率')
    plt.legend()
    plt.title(f'{model_name} 整体准确率曲线')
    
    plt.subplot(2, 2, 3)
    for dim in ['e_i', 'n_s', 't_f', 'j_p']:
        plt.plot(training_history[f'{dim}_acc'], label=dim)
    plt.xlabel('轮次')
    plt.ylabel('准确率')
    plt.legend()
    plt.title(f'{model_name} 各维度准确率曲线')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{model_name}_dimension_history.png')
    plt.close()
    
    # 加载最佳模型
    model.load_state_dict(torch.load(f'{OUTPUT_DIR}/{model_name}_best.pt'))
    
    return model, best_val_accuracy, val_dimension_accuracies

def train_traditional_models(X_train, y_train, X_val, y_val, feature_columns):
    """训练传统机器学习模型"""
    logger.info("训练传统机器学习模型...")
    
    # 提取特征
    X_train_features = X_train[feature_columns].values
    X_val_features = X_val[feature_columns].values
    
    # 创建TF-IDF特征
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train['advanced_posts'])
    X_val_tfidf = tfidf_vectorizer.transform(X_val['advanced_posts'])
    
    # 合并TF-IDF和特征
    from scipy.sparse import hstack, csr_matrix
    X_train_combined = hstack([X_train_tfidf, csr_matrix(X_train_features)])
    X_val_combined = hstack([X_val_tfidf, csr_matrix(X_val_features)])
    
    models = {}
    accuracies = {}
    
    # MBTI完整类型分类
    logger.info("训练SVM分类器用于完整MBTI分类...")
    svm_model = SVC(probability=True, random_state=RANDOM_SEED)
    svm_model.fit(X_train_combined, y_train['encoded_type'])
    y_val_pred = svm_model.predict(X_val_combined)
    accuracy = accuracy_score(y_val['encoded_type'], y_val_pred)
    logger.info(f"SVM完整类型分类器验证准确率: {accuracy:.4f}")
    models['svm_full'] = svm_model
    accuracies['svm_full'] = accuracy
    
    # 随机森林分类器
    logger.info("训练随机森林分类器用于完整MBTI分类...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    rf_model.fit(X_train_combined, y_train['encoded_type'])
    y_val_pred = rf_model.predict(X_val_combined)
    accuracy = accuracy_score(y_val['encoded_type'], y_val_pred)
    logger.info(f"随机森林完整类型分类器验证准确率: {accuracy:.4f}")
    models['rf_full'] = rf_model
    accuracies['rf_full'] = accuracy
    
    # 维度分类
    for dim in ['encoded_e_i', 'encoded_n_s', 'encoded_t_f', 'encoded_j_p']:
        dim_name = dim.split('_')[1]  # 获取简短名称
        
        logger.info(f"训练SVM分类器用于{dim_name}维度分类...")
        dim_svm = SVC(probability=True, random_state=RANDOM_SEED)
        dim_svm.fit(X_train_combined, y_train[dim])
        y_val_pred = dim_svm.predict(X_val_combined)
        accuracy = accuracy_score(y_val[dim], y_val_pred)
        logger.info(f"SVM {dim_name}维度分类器验证准确率: {accuracy:.4f}")
        models[f'svm_{dim_name}'] = dim_svm
        accuracies[f'svm_{dim_name}'] = accuracy
        
        logger.info(f"训练随机森林分类器用于{dim_name}维度分类...")
        dim_rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
        dim_rf.fit(X_train_combined, y_train[dim])
        y_val_pred = dim_rf.predict(X_val_combined)
        accuracy = accuracy_score(y_val[dim], y_val_pred)
        logger.info(f"随机森林 {dim_name}维度分类器验证准确率: {accuracy:.4f}")
        models[f'rf_{dim_name}'] = dim_rf
        accuracies[f'rf_{dim_name}'] = accuracy
    
    # 保存模型和向量化器
    joblib.dump(tfidf_vectorizer, f'{OUTPUT_DIR}/tfidf_vectorizer.joblib')
    for name, model in models.items():
        joblib.dump(model, f'{OUTPUT_DIR}/{name}_model.joblib')
    
    return {
        'models': models,
        'accuracies': accuracies,
        'tfidf_vectorizer': tfidf_vectorizer
    }

#################################################
# 评估和预测
#################################################

def evaluate_model(model, test_loader, criterion, label_encoder, device, model_name):
    """评估深度学习模型"""
    logger.info(f"评估 {model_name} 模型...")
    
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='测试中'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            outputs = model(input_ids, attention_mask)
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 计算准确率
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            # 收集预测结果
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
            
            # 更新测试损失
            test_loss += loss.item() * input_ids.size(0)
    
    test_loss /= test_total
    test_accuracy = test_correct / test_total
    
    logger.info(f'{model_name} 测试损失: {test_loss:.4f}, 测试准确率: {test_accuracy:.4f}')
    
    # 转换为MBTI类型
    pred_types = label_encoder.inverse_transform(all_preds)
    true_types = label_encoder.inverse_transform(all_labels)
    
    # 生成分类报告
    report = classification_report(true_types, pred_types)
    logger.info(f"\n{model_name} 分类报告:\n{report}")
    
    # 保存分类报告
    with open(f'{OUTPUT_DIR}/{model_name}_classification_report.txt', 'w') as f:
        f.write(report)
    
    # 计算每种MBTI类型的准确率
    type_accuracies = {}
    for mbti_type in np.unique(true_types):
        type_indices = np.where(true_types == mbti_type)[0]
        if len(type_indices) > 0:
            type_accuracy = np.mean(pred_types[type_indices] == true_types[type_indices])
            logger.info(f'{mbti_type} 准确率: {type_accuracy:.4f} (共 {len(type_indices)} 个样本)')
            type_accuracies[mbti_type] = type_accuracy
    
    # 绘制混淆矩阵
    plt.figure(figsize=(14, 12))
    cm = confusion_matrix(true_types, pred_types)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.xlabel('预测类型')
    plt.ylabel('真实类型')
    plt.title(f'{model_name} 混淆矩阵')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{model_name}_confusion_matrix.png')
    plt.close()
    
    return {
        'accuracy': test_accuracy,
        'loss': test_loss,
        'predictions': all_preds,
        'true_labels': all_labels,
        'probabilities': all_probs,
        'predicted_types': pred_types,
        'true_types': true_types,
        'type_accuracies': type_accuracies
    }

def evaluate_dimension_model(model, test_loader, criterion, dim_encoders, device, model_name):
    """评估维度分类模型"""
    logger.info(f"评估 {model_name} 维度模型...")
    
    model.eval()
    test_loss = 0.0
    test_total = 0
    dimension_correct = {'e_i': 0, 'n_s': 0, 't_f': 0, 'j_p': 0}
    dimension_preds = {'e_i': [], 'n_s': [], 't_f': [], 'j_p': []}
    dimension_labels = {'e_i': [], 'n_s': [], 't_f': [], 'j_p': []}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='测试维度'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 获取维度标签
            batch_labels = {
                'e_i': batch['e_i_label'].to(device) if 'e_i_label' in batch else None,
                'n_s': batch['n_s_label'].to(device) if 'n_s_label' in batch else None,
                't_f': batch['t_f_label'].to(device) if 't_f_label' in batch else None,
                'j_p': batch['j_p_label'].to(device) if 'j_p_label' in batch else None
            }
            
            # 前向传播
            outputs = model(input_ids, attention_mask)
            
            # 计算损失
            loss = 0
            for dim, labels in batch_labels.items():
                if labels is not None:
                    dim_loss = criterion(outputs[dim], labels)
                    loss += dim_loss
            
            # 计算准确率
            batch_size = input_ids.size(0)
            test_total += batch_size
            
            for dim, labels in batch_labels.items():
                if labels is not None:
                    _, predicted = torch.max(outputs[dim], 1)
                    dimension_correct[dim] += (predicted == labels).sum().item()
                    
                    # 收集预测和标签
                    dimension_preds[dim].extend(predicted.cpu().numpy())
                    dimension_labels[dim].extend(labels.cpu().numpy())
            
            # 更新测试损失
            test_loss += loss.item() * batch_size
    
    test_loss /= test_total
    dimension_accuracies = {dim: dimension_correct[dim] / test_total 
                          for dim in dimension_correct}
    test_accuracy = sum(dimension_accuracies.values()) / len(dimension_accuracies)
    
    logger.info(f'{model_name} 测试损失: {test_loss:.4f}, 测试准确率: {test_accuracy:.4f}')
    logger.info(f'维度准确率: {", ".join([f"{dim}: {acc:.4f}" for dim, acc in dimension_accuracies.items()])}')
    
    # 为每个维度生成分类报告
    dimension_reports = {}
    for dim in dimension_preds:
        if len(dimension_preds[dim]) > 0:
            # 转换为实际维度值
            pred_values = dim_encoders[dim].inverse_transform(dimension_preds[dim])
            true_values = dim_encoders[dim].inverse_transform(dimension_labels[dim])
            
            # 生成分类报告
            report = classification_report(true_values, pred_values)
            logger.info(f"\n{model_name} {dim} 维度分类报告:\n{report}")
            dimension_reports[dim] = report
            
            # 保存分类报告
            with open(f'{OUTPUT_DIR}/{model_name}_{dim}_classification_report.txt', 'w') as f:
                f.write(report)
            
            # 绘制混淆矩阵
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(true_values, pred_values)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=dim_encoders[dim].classes_, 
                        yticklabels=dim_encoders[dim].classes_)
            plt.xlabel('预测值')
            plt.ylabel('真实值')
            plt.title(f'{model_name} {dim} 维度混淆矩阵')
            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/{model_name}_{dim}_confusion_matrix.png')
            plt.close()
    
    return {
        'accuracy': test_accuracy,
        'loss': test_loss,
        'dimension_accuracies': dimension_accuracies,
        'dimension_predictions': dimension_preds,
        'dimension_labels': dimension_labels,
        'dimension_reports': dimension_reports
    }

def evaluate_traditional_models(models, X_test, y_test, tfidf_vectorizer, 
                              feature_columns, label_encoder, dim_encoders):
    """评估传统机器学习模型"""
    logger.info("评估传统机器学习模型...")
    
    # 提取特征
    X_test_features = X_test[feature_columns].values
    
    # 创建TF-IDF特征
    X_test_tfidf = tfidf_vectorizer.transform(X_test['advanced_posts'])
    
    # 合并TF-IDF和特征
    from scipy.sparse import hstack, csr_matrix
    X_test_combined = hstack([X_test_tfidf, csr_matrix(X_test_features)])
    
    results = {}
    
    # 评估完整MBTI分类器
    for model_name in ['svm_full', 'rf_full']:
        if model_name in models:
            model = models[model_name]
            y_pred = model.predict(X_test_combined)
            accuracy = accuracy_score(y_test['encoded_type'], y_pred)
            
            # 转换为MBTI类型
            pred_types = label_encoder.inverse_transform(y_pred)
            true_types = label_encoder.inverse_transform(y_test['encoded_type'])
            
            # 生成分类报告
            report = classification_report(true_types, pred_types)
            logger.info(f"\n{model_name} 分类报告:\n{report}")
            
            # 保存分类报告
            with open(f'{OUTPUT_DIR}/{model_name}_classification_report.txt', 'w') as f:
                f.write(report)
            
            # 计算每种MBTI类型的准确率
            type_accuracies = {}
            for mbti_type in np.unique(true_types):
                type_indices = np.where(true_types == mbti_type)[0]
                if len(type_indices) > 0:
                    type_accuracy = np.mean(pred_types[type_indices] == true_types[type_indices])
                    logger.info(f'{model_name} - {mbti_type} 准确率: {type_accuracy:.4f} (共 {len(type_indices)} 个样本)')
                    type_accuracies[mbti_type] = type_accuracy
            
            # 绘制混淆矩阵
            plt.figure(figsize=(14, 12))
            cm = confusion_matrix(true_types, pred_types)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=label_encoder.classes_, 
                        yticklabels=label_encoder.classes_)
            plt.xlabel('预测类型')
            plt.ylabel('真实类型')
            plt.title(f'{model_name} 混淆矩阵')
            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/{model_name}_confusion_matrix.png')
            plt.close()
            
            results[model_name] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'predicted_types': pred_types,
                'true_types': true_types,
                'type_accuracies': type_accuracies,
                'classification_report': report
            }
    
    # 评估维度分类器
    for dim in ['e_i', 'n_s', 't_f', 'j_p']:
        for model_type in ['svm', 'rf']:
            model_name = f'{model_type}_{dim}'
            if model_name in models:
                model = models[model_name]
                encoded_dim = f'encoded_{dim}'
                y_pred = model.predict(X_test_combined)
                accuracy = accuracy_score(y_test[encoded_dim], y_pred)
                
                # 转换为维度值
                pred_values = dim_encoders[dim].inverse_transform(y_pred)
                true_values = dim_encoders[dim].inverse_transform(y_test[encoded_dim])
                
                # 生成分类报告
                report = classification_report(true_values, pred_values)
                logger.info(f"\n{model_name} 分类报告:\n{report}")
                
                # 保存分类报告
                with open(f'{OUTPUT_DIR}/{model_name}_classification_report.txt', 'w') as f:
                    f.write(report)
                
                # 绘制混淆矩阵
                plt.figure(figsize=(8, 6))
                cm = confusion_matrix(true_values, pred_values)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=dim_encoders[dim].classes_, 
                            yticklabels=dim_encoders[dim].classes_)
                plt.xlabel('预测值')
                plt.ylabel('真实值')
                plt.title(f'{model_name} 混淆矩阵')
                plt.tight_layout()
                plt.savefig(f'{OUTPUT_DIR}/{model_name}_confusion_matrix.png')
                plt.close()
                
                results[model_name] = {
                    'accuracy': accuracy,
                    'predictions': y_pred,
                    'predicted_values': pred_values,
                    'true_values': true_values,
                    'classification_report': report
                }
    
    return results

def ensemble_prediction(text, dl_models, trad_models, vocab, processor, 
                       tfidf_vectorizer, label_encoder, dim_encoders, 
                       feature_columns, device, model_weights=None):
    """使用模型集成进行MBTI预测"""
    # 清理和处理文本
    cleaned_text = processor.basic_clean(text)
    advanced_text = processor.advanced_clean(text)
    
    # 提取特征
    text_features = processor.extract_features(text)
    
    # 初始化预测结果存储
    model_predictions = {}
    model_probabilities = {}
    dimension_predictions = {'e_i': {}, 'n_s': {}, 't_f': {}, 'j_p': {}}
    
    # 1. 使用深度学习全类型模型预测
    for model_name, model in dl_models.items():
        if 'dimension' not in model_name:  # 仅完整类型分类器
            # 分词
            tokens = processor.tokenize(cleaned_text)
            
            # 转换为索引
            indices = [vocab[token] for token in tokens]
            
            # 截断或填充
            max_length = 512
            if len(indices) > max_length:
                indices = indices[:max_length]
            else:
                indices = indices + [vocab["<pad>"]] * (max_length - len(indices))
            
            # 创建掩码
            mask = torch.zeros(max_length, dtype=torch.long)
            mask[:min(len(tokens), max_length)] = 1
            
            # 转换为张量
            input_ids = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
            attention_mask = mask.unsqueeze(0).to(device)
            
            # 预测
            model.eval()
            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                top_p, top_class = torch.topk(probabilities, 3)
                
                # 转换为MBTI类型
                pred_type = label_encoder.inverse_transform([top_class[0].item()])[0]
                model_predictions[model_name] = pred_type
                model_probabilities[model_name] = probabilities.cpu().numpy()
    
    # 2. 使用深度学习维度模型预测
    for model_name, model in dl_models.items():
        if 'dimension' in model_name:  # 仅维度分类器
            # 分词
            tokens = processor.tokenize(cleaned_text)
            
            # 转换为索引
            indices = [vocab[token] for token in tokens]
            
            # 截断或填充
            max_length = 512
            if len(indices) > max_length:
                indices = indices[:max_length]
            else:
                indices = indices + [vocab["<pad>"]] * (max_length - len(indices))
            
            # 创建掩码
            mask = torch.zeros(max_length, dtype=torch.long)
            mask[:min(len(tokens), max_length)] = 1
            
            # 转换为张量
            input_ids = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
            attention_mask = mask.unsqueeze(0).to(device)
            
            # 预测
            model.eval()
            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
                
                # 获取每个维度的预测
                dimension_values = {}
                for dim in ['e_i', 'n_s', 't_f', 'j_p']:
                    dim_probs = torch.nn.functional.softmax(outputs[dim], dim=1)[0]
                    dim_pred = torch.argmax(dim_probs).item()
                    dim_value = dim_encoders[dim].inverse_transform([dim_pred])[0]
                    dimension_values[dim] = dim_value
                    dimension_predictions[dim][model_name] = dim_value
                
                # 组合为完整MBTI类型
                combined_type = ''.join([dimension_values[dim] for dim in ['e_i', 'n_s', 't_f', 'j_p']])
                model_predictions[model_name] = combined_type
    
    # 3. 使用传统机器学习模型预测
    if trad_models and tfidf_vectorizer:
        # 创建特征向量
        features_array = np.array([[text_features[col] for col in feature_columns]])
        text_tfidf = tfidf_vectorizer.transform([advanced_text])
        
        # 合并TF-IDF和特征
        from scipy.sparse import hstack, csr_matrix
        features_combined = hstack([text_tfidf, csr_matrix(features_array)])
        
        # 完整类型预测
        for model_name in ['svm_full', 'rf_full']:
            if model_name in trad_models:
                model = trad_models[model_name]
                pred_idx = model.predict(features_combined)[0]
                pred_type = label_encoder.inverse_transform([pred_idx])[0]
                model_predictions[model_name] = pred_type
                
                # 获取概率
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(features_combined)[0]
                    model_probabilities[model_name] = probs
        
        # 维度预测
        for dim in ['e_i', 'n_s', 't_f', 'j_p']:
            for model_type in ['svm', 'rf']:
                model_name = f'{model_type}_{dim}'
                if model_name in trad_models:
                    model = trad_models[model_name]
                    dim_pred = model.predict(features_combined)[0]
                    dim_value = dim_encoders[dim].inverse_transform([dim_pred])[0]
                    dimension_predictions[dim][model_name] = dim_value
        
        # 组合传统模型的维度预测
        if all(len(dimension_predictions[dim]) >= 2 for dim in dimension_predictions):
            svm_dimensions = [dimension_predictions[dim].get('svm_' + dim, None) for dim in ['e_i', 'n_s', 't_f', 'j_p']]
            rf_dimensions = [dimension_predictions[dim].get('rf_' + dim, None) for dim in ['e_i', 'n_s', 't_f', 'j_p']]
            
            if None not in svm_dimensions:
                model_predictions['svm_dimension'] = ''.join(svm_dimensions)
            
            if None not in rf_dimensions:
                model_predictions['rf_dimension'] = ''.join(rf_dimensions)
    
    # 4. 集成预测结果
    # 模型权重（可以基于验证集准确率或先验知识设置）
    if model_weights is None:
        model_weights = {model: 1.0 for model in model_predictions}
    
    # 集成投票
    type_votes = {}
    for model, pred_type in model_predictions.items():
        weight = model_weights.get(model, 1.0)
        
        if pred_type not in type_votes:
            type_votes[pred_type] = 0
        
        type_votes[pred_type] += weight
    
    # 获取投票最多的类型
    if type_votes:
        ensemble_type = max(type_votes.items(), key=lambda x: x[1])[0]
        ensemble_confidence = type_votes[ensemble_type] / sum(type_votes.values())
    else:
        # 如果没有预测，返回默认值
        ensemble_type = "UNKNOWN"
        ensemble_confidence = 0.0
    
    # 5. 获取维度投票结果
    dimension_results = {}
    for dim in dimension_predictions:
        dim_votes = {}
        for model, value in dimension_predictions[dim].items():
            weight = model_weights.get(model, 1.0)
            
            if value not in dim_votes:
                dim_votes[value] = 0
            
            dim_votes[value] += weight
        
        if dim_votes:
            dimension_results[dim] = max(dim_votes.items(), key=lambda x: x[1])[0]
        else:
            dimension_results[dim] = "?"
    
    # 组合维度结果
    dimension_combined = ''.join([dimension_results.get(dim, "?") for dim in ['e_i', 'n_s', 't_f', 'j_p']])
    
    # 获取排序后的备选预测
    sorted_votes = sorted(type_votes.items(), key=lambda x: x[1], reverse=True)
    alternative_types = [(t, v/sum(type_votes.values())) for t, v in sorted_votes[1:4]] if len(sorted_votes) > 1 else []
    
    return {
        'ensemble_type': ensemble_type,
        'confidence': ensemble_confidence,
        'dimension_results': dimension_results,
        'dimension_combined': dimension_combined,
        'model_predictions': model_predictions,
        'alternative_types': alternative_types
    }

#################################################
# 训练和评估脚本
#################################################

def train_all_models(file_path, output_dir=OUTPUT_DIR, epochs=8, batch_size=16):
    """训练所有模型并进行集成评估"""
    try:
        # 1. 数据处理
        processor = TextProcessor()
        data = load_and_process_data(file_path, processor)
        
        # 提取需要的数据
        X_train = data['X_train']
        X_val = data['X_val']
        X_test = data['X_test']
        y_train = data['y_train']
        y_val = data['y_val']
        y_test = data['y_test']
        label_encoder = data['label_encoder']
        dim_encoders = data['dim_encoders']
        class_weights = data['class_weights']
        num_classes = data['num_classes']
        vocab = data['vocab']
        
        # 保存词汇表和编码器
        joblib.dump(vocab, f'{output_dir}/vocabulary.joblib')
        joblib.dump(label_encoder, f'{output_dir}/label_encoder.joblib')
        for dim, encoder in dim_encoders.items():
            joblib.dump(encoder, f'{output_dir}/dimension_encoder_{dim}.joblib')
        
        # 特征列
        feature_columns = [col for col in X_train.columns if col not in ['cleaned_posts', 'advanced_posts']]
        
        # 2. 训练深度学习模型
        dl_models = {}
        model_accuracies = {}
        
        # 创建权重张量用于损失函数
        weight_tensor = torch.tensor([class_weights[label_encoder.inverse_transform([i])[0]] 
                                    for i in range(num_classes)]).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        
        # 2.1 训练LSTM模型
        logger.info("\n=== 训练LSTM模型 ===")
        
        # 创建数据集
        train_dataset = MBTIDataset(
            X_train['cleaned_posts'].tolist(), 
            y_train['encoded_type'].tolist(), 
            vocab, 
            processor.tokenize
        )
        val_dataset = MBTIDataset(
            X_val['cleaned_posts'].tolist(), 
            y_val['encoded_type'].tolist(), 
            vocab, 
            processor.tokenize
        )
        test_dataset = MBTIDataset(
            X_test['cleaned_posts'].tolist(), 
            y_test['encoded_type'].tolist(), 
            vocab, 
            processor.tokenize
        )
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # 初始化模型
        embedding_dim = 300
        hidden_dim = 256
        lstm_model = LSTMMBTIClassifier(
            vocab_size=len(vocab), 
            embedding_dim=embedding_dim, 
            hidden_dim=hidden_dim, 
            num_classes=num_classes
        )
        lstm_model = lstm_model.to(device)
        
        # 优化器
        optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
        
        # 训练
        lstm_model, lstm_accuracy = train_model(
            model=lstm_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=epochs,
            device=device,
            model_name='lstm_full'
        )
        
        dl_models['lstm_full'] = lstm_model
        model_accuracies['lstm_full'] = lstm_accuracy
        
        # 2.2 训练CNN模型
        logger.info("\n=== 训练CNN模型 ===")
        
        # 初始化模型
        cnn_model = CNNMBTIClassifier(
            vocab_size=len(vocab), 
            embedding_dim=embedding_dim, 
            num_classes=num_classes
        )
        cnn_model = cnn_model.to(device)
        
        # 优化器
        optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
        
        # 训练
        cnn_model, cnn_accuracy = train_model(
            model=cnn_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=epochs,
            device=device,
            model_name='cnn_full'
        )
        
        dl_models['cnn_full'] = cnn_model
        model_accuracies['cnn_full'] = cnn_accuracy
        
        # 2.3 训练GRU模型
        logger.info("\n=== 训练GRU模型 ===")
        
        # 初始化模型
        gru_model = GRUMBTIClassifier(
            vocab_size=len(vocab), 
            embedding_dim=embedding_dim, 
            hidden_dim=hidden_dim, 
            num_classes=num_classes
        )
        gru_model = gru_model.to(device)
        
        # 优化器
        optimizer = optim.Adam(gru_model.parameters(), lr=0.001)
        
        # 训练
        gru_model, gru_accuracy = train_model(
            model=gru_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=epochs,
            device=device,
            model_name='gru_full'
        )
        
        dl_models['gru_full'] = gru_model
        model_accuracies['gru_full'] = gru_accuracy
        
        # 2.4 训练混合模型
        logger.info("\n=== 训练混合模型 ===")
        
        # 初始化模型
        hybrid_model = HybridMBTIClassifier(
            vocab_size=len(vocab), 
            embedding_dim=embedding_dim, 
            hidden_dim=hidden_dim, 
            num_classes=num_classes
        )
        hybrid_model = hybrid_model.to(device)
        
        # 优化器
        optimizer = optim.Adam(hybrid_model.parameters(), lr=0.001)
        
        # 训练
        hybrid_model, hybrid_accuracy = train_model(
            model=hybrid_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=epochs,
            device=device,
            model_name='hybrid_full'
        )
        
        dl_models['hybrid_full'] = hybrid_model
        model_accuracies['hybrid_full'] = hybrid_accuracy
        
        # 3. 训练维度分解模型
        logger.info("\n=== 训练维度分解模型 ===")
        
        # 创建维度标签
        dimension_labels = {
            'e_i': {'e_i_label': y_train['encoded_e_i'].tolist()},
            'n_s': {'n_s_label': y_train['encoded_n_s'].tolist()},
            't_f': {'t_f_label': y_train['encoded_t_f'].tolist()},
            'j_p': {'j_p_label': y_train['encoded_j_p'].tolist()}
        }
        
        val_dimension_labels = {
            'e_i': {'e_i_label': y_val['encoded_e_i'].tolist()},
            'n_s': {'n_s_label': y_val['encoded_n_s'].tolist()},
            't_f': {'t_f_label': y_val['encoded_t_f'].tolist()},
            'j_p': {'j_p_label': y_val['encoded_j_p'].tolist()}
        }
        
        test_dimension_labels = {
            'e_i': {'e_i_label': y_test['encoded_e_i'].tolist()},
            'n_s': {'n_s_label': y_test['encoded_n_s'].tolist()},
            't_f': {'t_f_label': y_test['encoded_t_f'].tolist()},
            'j_p': {'j_p_label': y_test['encoded_j_p'].tolist()}
        }
        
        # 3.1 创建数据集
        class DimensionDatasetWrapper(Dataset):
            def __init__(self, texts, dim_labels, vocab, tokenizer, max_length=512):
                self.texts = texts
                self.tokenizer = tokenizer
                self.vocab = vocab
                self.max_length = max_length
                self.labels = dim_labels
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                text = self.texts[idx]
                
                # 分词
                tokens = self.tokenizer(text)
                
                # 转换为索引
                indices = [self.vocab[token] for token in tokens]
                
                # 截断或填充
                if len(indices) > self.max_length:
                    indices = indices[:self.max_length]
                else:
                    indices = indices + [self.vocab["<pad>"]] * (self.max_length - len(indices))
                
                # 转换为张量
                indices_tensor = torch.tensor(indices, dtype=torch.long)
                
                # 创建掩码
                mask = torch.zeros(self.max_length, dtype=torch.long)
                mask[:min(len(tokens), self.max_length)] = 1
                
                # 准备结果字典
                result = {
                    'input_ids': indices_tensor,
                    'attention_mask': mask
                }
                
                # 添加各维度标签
                for dim, labels_dict in self.labels.items():
                    for label_name, labels in labels_dict.items():
                        result[label_name] = torch.tensor(labels[idx], dtype=torch.long)
                
                return result
        
        # 创建训练集
        train_dim_dataset = DimensionDatasetWrapper(
            X_train['cleaned_posts'].tolist(),
            dimension_labels,
            vocab,
            processor.tokenize
        )
        
        # 创建验证集
        val_dim_dataset = DimensionDatasetWrapper(
            X_val['cleaned_posts'].tolist(),
            val_dimension_labels,
            vocab,
            processor.tokenize
        )
        
        # 创建测试集
        test_dim_dataset = DimensionDatasetWrapper(
            X_test['cleaned_posts'].tolist(),
            test_dimension_labels,
            vocab,
            processor.tokenize
        )
        
        # 创建数据加载器
        train_dim_loader = DataLoader(train_dim_dataset, batch_size=batch_size, shuffle=True)
        val_dim_loader = DataLoader(val_dim_dataset, batch_size=batch_size)
        test_dim_loader = DataLoader(test_dim_dataset, batch_size=batch_size)
        
        # 3.1 LSTM维度模型
        logger.info("\n=== 训练LSTM维度模型 ===")
        
        # 初始化模型
        lstm_dim_model = DimensionClassifier(
            vocab_size=len(vocab),
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            model_type='lstm'
        )
        lstm_dim_model = lstm_dim_model.to(device)
        
        # 优化器
        optimizer = optim.Adam(lstm_dim_model.parameters(), lr=0.001)
        
        # 训练
        lstm_dim_model, lstm_dim_accuracy, lstm_dim_accuracies = train_dimension_model(
            model=lstm_dim_model,
            train_loader=train_dim_loader,
            val_loader=val_dim_loader,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            num_epochs=epochs,
            device=device,
            model_name='lstm_dimension'
        )
        
        dl_models['lstm_dimension'] = lstm_dim_model
        model_accuracies['lstm_dimension'] = lstm_dim_accuracy
        
        # 3.2 CNN维度模型
        logger.info("\n=== 训练CNN维度模型 ===")
        
        # 初始化模型
        cnn_dim_model = DimensionClassifier(
            vocab_size=len(vocab),
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            model_type='cnn'
        )
        cnn_dim_model = cnn_dim_model.to(device)
        
        # 优化器
        optimizer = optim.Adam(cnn_dim_model.parameters(), lr=0.001)
        
        # 训练
        cnn_dim_model, cnn_dim_accuracy, cnn_dim_accuracies = train_dimension_model(
            model=cnn_dim_model,
            train_loader=train_dim_loader,
            val_loader=val_dim_loader,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            num_epochs=epochs,
            device=device,
            model_name='cnn_dimension'
        )
        
        dl_models['cnn_dimension'] = cnn_dim_model
        model_accuracies['cnn_dimension'] = cnn_dim_accuracy
        
        # 4. 训练传统机器学习模型
        logger.info("\n=== 训练传统机器学习模型 ===")
        trad_results = train_traditional_models(
            X_train, y_train, X_val, y_val, feature_columns
        )
        
        trad_models = trad_results['models']
        trad_accuracies = trad_results['accuracies']
        tfidf_vectorizer = trad_results['tfidf_vectorizer']
        
        # 合并所有准确率
        for model_name, accuracy in trad_accuracies.items():
            model_accuracies[model_name] = accuracy
        
        # 5. 评估模型
        logger.info("\n=== 评估模型 ===")
        
        # 5.1 评估深度学习完整类型模型
        dl_results = {}
        for model_name, model in dl_models.items():
            if 'dimension' not in model_name:  # 仅评估完整类型模型
                # 评估
                result = evaluate_model(
                    model=model,
                    test_loader=test_loader,
                    criterion=criterion,
                    label_encoder=label_encoder,
                    device=device,
                    model_name=model_name
                )
                
                dl_results[model_name] = result
        
        # 5.2 评估维度模型
        dimension_results = {}
        for model_name, model in dl_models.items():
            if 'dimension' in model_name:  # 仅评估维度模型
                # 评估
                result = evaluate_dimension_model(
                    model=model,
                    test_loader=test_dim_loader,
                    criterion=nn.CrossEntropyLoss(),
                    dim_encoders=dim_encoders,
                    device=device,
                    model_name=model_name
                )
                
                dimension_results[model_name] = result
        
        # 5.3 评估传统机器学习模型
        trad_eval_results = evaluate_traditional_models(
            trad_models, X_test, y_test, tfidf_vectorizer, 
            feature_columns, label_encoder, dim_encoders
        )
        
        # 6. 集成评估
        logger.info("\n=== 集成评估 ===")
        
        # 根据验证准确率为每个模型分配权重
        model_weights = {model: accuracy for model, accuracy in model_accuracies.items()}
        
        # 为维度模型的每个维度分量分配权重
        for model_name in dimension_results:
            for dim, acc in dimension_results[model_name]['dimension_accuracies'].items():
                model_weights[f'{model_name}_{dim}'] = acc
        
        # 评估集成准确率
        ensemble_correct = 0
        ensemble_total = 0
        ensemble_predictions = []
        dimension_ensemble_correct = {'e_i': 0, 'n_s': 0, 't_f': 0, 'j_p': 0}
        dimension_ensemble_total = {'e_i': 0, 'n_s': 0, 't_f': 0, 'j_p': 0}
        
        for i in range(len(X_test)):
            # 获取样本文本
            text = X_test['cleaned_posts'].iloc[i]
            
            # 进行集成预测
            prediction = ensemble_prediction(
                text=text,
                dl_models=dl_models,
                trad_models=trad_models,
                vocab=vocab,
                processor=processor,
                tfidf_vectorizer=tfidf_vectorizer,
                label_encoder=label_encoder,
                dim_encoders=dim_encoders,
                feature_columns=feature_columns,
                device=device,
                model_weights=model_weights
            )
            
            # 获取真实标签
            true_type = label_encoder.inverse_transform([y_test['encoded_type'].iloc[i]])[0]
            
            # 统计准确率
            if prediction['ensemble_type'] == true_type:
                ensemble_correct += 1
            
            ensemble_total += 1
            ensemble_predictions.append(prediction)
            
            # 统计维度准确率
            for dim in ['e_i', 'n_s', 't_f', 'j_p']:
                true_dim = y_test[dim].iloc[i]
                pred_dim = prediction['dimension_results'].get(dim, None)
                
                if pred_dim == true_dim:
                    dimension_ensemble_correct[dim] += 1
                
                dimension_ensemble_total[dim] += 1
        
        # 计算集成准确率
        ensemble_accuracy = ensemble_correct / ensemble_total
        dimension_ensemble_accuracies = {dim: dimension_ensemble_correct[dim] / dimension_ensemble_total[dim]
                                       for dim in dimension_ensemble_correct}
        
        logger.info(f"集成模型准确率: {ensemble_accuracy:.4f}")
        logger.info(f"维度集成准确率: {', '.join([f'{dim}: {acc:.4f}' for dim, acc in dimension_ensemble_accuracies.items()])}")
        
        # 7. 保存模型和结果
        logger.info("\n=== 保存模型和结果 ===")
        
        # 保存集成元数据
        ensemble_metadata = {
            'model_weights': model_weights,
            'ensemble_accuracy': ensemble_accuracy,
            'dimension_ensemble_accuracies': dimension_ensemble_accuracies,
            'model_accuracies': model_accuracies
        }
        
        joblib.dump(ensemble_metadata, f'{output_dir}/ensemble_metadata.joblib')
        
        # 保存词汇表
        joblib.dump(vocab, f'{output_dir}/vocab.joblib')
        
        # 返回训练结果
        return {
            'dl_models': dl_models,
            'trad_models': trad_models,
            'vocab': vocab,
            'tfidf_vectorizer': tfidf_vectorizer,
            'label_encoder': label_encoder,
            'dim_encoders': dim_encoders,
            'processor': processor,
            'feature_columns': feature_columns,
            'model_weights': model_weights,
            'ensemble_accuracy': ensemble_accuracy,
            'dl_results': dl_results,
            'dimension_results': dimension_results,
            'trad_eval_results': trad_eval_results,
            'X_test': X_test,
            'y_test': y_test
        }
    
    except Exception as e:
        logger.error(f"训练模型时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def load_ensemble_model(model_dir=OUTPUT_DIR):
    """加载保存的集成模型"""
    try:
        # 加载元数据
        ensemble_metadata = joblib.load(f'{model_dir}/ensemble_metadata.joblib')
        model_weights = ensemble_metadata['model_weights']
        
        # 加载标签编码器
        label_encoder = joblib.load(f'{model_dir}/label_encoder.joblib')
        
        # 加载维度编码器
        dim_encoders = {}
        for dim in ['e_i', 'n_s', 't_f', 'j_p']:
            dim_encoders[dim] = joblib.load(f'{model_dir}/dimension_encoder_{dim}.joblib')
        
        # 加载词汇表
        vocab = joblib.load(f'{model_dir}/vocab.joblib')
        
        # 加载文本处理器
        processor = TextProcessor()
        
        # 加载深度学习模型
        dl_models = {}
        
        # 加载LSTM完整类型模型
        if os.path.exists(f'{model_dir}/lstm_full_best.pt'):
            lstm_model = LSTMMBTIClassifier(
                vocab_size=len(vocab),
                embedding_dim=300,
                hidden_dim=256,
                num_classes=len(label_encoder.classes_)
            )
            lstm_model.load_state_dict(torch.load(f'{model_dir}/lstm_full_best.pt', map_location=device))
            lstm_model = lstm_model.to(device)
            lstm_model.eval()
            dl_models['lstm_full'] = lstm_model
        
        # 加载CNN完整类型模型
        if os.path.exists(f'{model_dir}/cnn_full_best.pt'):
            cnn_model = CNNMBTIClassifier(
                vocab_size=len(vocab),
                embedding_dim=300,
                num_classes=len(label_encoder.classes_)
            )
            cnn_model.load_state_dict(torch.load(f'{model_dir}/cnn_full_best.pt', map_location=device))
            cnn_model = cnn_model.to(device)
            cnn_model.eval()
            dl_models['cnn_full'] = cnn_model
        
        # 加载GRU完整类型模型
        if os.path.exists(f'{model_dir}/gru_full_best.pt'):
            gru_model = GRUMBTIClassifier(
                vocab_size=len(vocab),
                embedding_dim=300,
                hidden_dim=256,
                num_classes=len(label_encoder.classes_)
            )
            gru_model.load_state_dict(torch.load(f'{model_dir}/gru_full_best.pt', map_location=device))
            gru_model = gru_model.to(device)
            gru_model.eval()
            dl_models['gru_full'] = gru_model
        
        # 加载混合模型
        if os.path.exists(f'{model_dir}/hybrid_full_best.pt'):
            hybrid_model = HybridMBTIClassifier(
                vocab_size=len(vocab),
                embedding_dim=300,
                hidden_dim=256,
                num_classes=len(label_encoder.classes_)
            )
            hybrid_model.load_state_dict(torch.load(f'{model_dir}/hybrid_full_best.pt', map_location=device))
            hybrid_model = hybrid_model.to(device)
            hybrid_model.eval()
            dl_models['hybrid_full'] = hybrid_model
        
        # 加载维度模型
        if os.path.exists(f'{model_dir}/lstm_dimension_best.pt'):
            lstm_dim_model = DimensionClassifier(
                vocab_size=len(vocab),
                embedding_dim=300,
                hidden_dim=256,
                model_type='lstm'
            )
            lstm_dim_model.load_state_dict(torch.load(f'{model_dir}/lstm_dimension_best.pt', map_location=device))
            lstm_dim_model = lstm_dim_model.to(device)
            lstm_dim_model.eval()
            dl_models['lstm_dimension'] = lstm_dim_model
        
        if os.path.exists(f'{model_dir}/cnn_dimension_best.pt'):
            cnn_dim_model = DimensionClassifier(
                vocab_size=len(vocab),
                embedding_dim=300,
                hidden_dim=256,
                model_type='cnn'
            )
            cnn_dim_model.load_state_dict(torch.load(f'{model_dir}/cnn_dimension_best.pt', map_location=device))
            cnn_dim_model = cnn_dim_model.to(device)
            cnn_dim_model.eval()
            dl_models['cnn_dimension'] = cnn_dim_model
        
        # 加载传统机器学习模型
        trad_models = {}
        model_files = [f for f in os.listdir(model_dir) if f.endswith('_model.joblib')]
        for file in model_files:
            model_name = file.replace('_model.joblib', '')
            trad_models[model_name] = joblib.load(f'{model_dir}/{file}')
        
        # 加载TF-IDF向量化器
        tfidf_vectorizer = None
        if os.path.exists(f'{model_dir}/tfidf_vectorizer.joblib'):
            tfidf_vectorizer = joblib.load(f'{model_dir}/tfidf_vectorizer.joblib')
        
        # 特征列
        feature_columns = ['text_length', 'sentence_count', 'word_count', 'avg_word_length', 'sentiment', 'subjectivity',
                         'question_ratio', 'exclamation_ratio', 'uppercase_ratio']
        
        logger.info(f"成功从 {model_dir} 加载集成模型")
        
        return {
            'dl_models': dl_models,
            'trad_models': trad_models,
            'vocab': vocab,
            'tfidf_vectorizer': tfidf_vectorizer,
            'label_encoder': label_encoder,
            'dim_encoders': dim_encoders,
            'processor': processor,
            'feature_columns': feature_columns,
            'model_weights': model_weights
        }
    
    except Exception as e:
        logger.error(f"加载集成模型时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def predict_mbti_type(text, models=None, model_dir=OUTPUT_DIR):
    """预测文本的MBTI类型"""
    try:
        # 如果没有提供模型，则加载模型
        if models is None:
            models = load_ensemble_model(model_dir)
        
        # 进行集成预测
        prediction = ensemble_prediction(
            text=text,
            dl_models=models['dl_models'],
            trad_models=models['trad_models'],
            vocab=models['vocab'],
            processor=models['processor'],
            tfidf_vectorizer=models['tfidf_vectorizer'],
            label_encoder=models['label_encoder'],
            dim_encoders=models['dim_encoders'],
            feature_columns=models['feature_columns'],
            device=device,
            model_weights=models['model_weights']
        )
        
        # 返回预测结果
        return prediction
    
    except Exception as e:
        logger.error(f"预测MBTI类型时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'ensemble_type': "ERROR",
            'error': str(e)
        }

def explain_mbti_type(mbti_type):
    """解释MBTI类型的特点"""
    # MBTI类型说明
    type_descriptions = {
        'ISTJ': {
            'name': '检查员',
            'description': '安静、严肃、通过全面性和可靠性获得成功。实际、注重事实、负责任。通过合理地决定要做什么，不顾抗议和干扰坚定地去做。以条理、有序、系统的方式认真生活和工作。重视传统和忠诚。',
            'strengths': ['负责任', '实际可靠', '忠诚', '注重细节', '有条理'],
            'weaknesses': ['可能过于严格', '不易接受变化', '可能固执', '易忽视情感因素', '可能判断太快']
        },
        'ISFJ': {
            'name': '守卫者',
            'description': '安静、友好、负责任和尽职尽责。工作勤奋，达到其承诺。忠诚、体贴、关注他人感受、关心他人的感受。努力创造和谐有序的工作和家庭环境。',
            'strengths': ['忠诚', '体贴', '注重细节', '耐心', '责任感强'],
            'weaknesses': ['过度牺牲自我', '难以说"不"', '可能容易受伤', '抗拒变化', '可能忽视自身需求']
        },
        'INFJ': {
            'name': '倡导者',
            'description': '寻求意义和联系，了解人与人之间的动机。有较强的洞察力。有能力关心别人。有条理和果断地实现自己的见解。注重终身学习。',
            'strengths': ['富有洞察力', '理想主义', '创造力强', '坚定不移', '富有同情心'],
            'weaknesses': ['完美主义', '对批评敏感', '非常私密', '容易倦怠', '有时过于理想化']
        },
        'INTJ': {
            'name': '建筑师',
            'description': '对自己的思想和智力能力有强烈的原创性和动力。对他们的知识和能力充满信心。能够独立工作，思考未来，期望结果。对于实现其想法和目标有组织有决心。',
            'strengths': ['有战略思维', '独立', '坚定', '分析能力强', '有创新思维'],
            'weaknesses': ['可能过于批判', '过度独立', '情感上疏离', '固执', '完美主义']
        },
        'ISTP': {
            'name': '鉴赏家',
            'description': '宽容、灵活，沉默地观察直到问题出现，然后迅速行动找到可行的解决方案。分析事物如何运作，并可以通过大量的信息迅速找到问题的核心。对原因和结果感兴趣，以逻辑的方式组织事实，重视效率。',
            'strengths': ['灵活适应力强', '实用主义', '独立', '解决问题的能力强', '冷静沉着'],
            'weaknesses': ['回避承诺', '对情感反应迟钝', '容易感到厌倦', '冒险倾向', '不耐烦']
        },
        'ISFP': {
            'name': '探险家',
            'description': '安静、友好、敏感和善良。享受当下，喜欢自己的空间，按照自己的时间做事。忠诚、坚定地致力于自己重视的人或事。不喜欢不和和冲突，不把自己的观点或价值观强加于人。',
            'strengths': ['善良敏感', '艺术气质', '忠诚', '灵活', '关注当下'],
            'weaknesses': ['过度害羞', '避免冲突', '对批评敏感', '过于自我保护', '计划性不足']
        },
        'INFP': {
            'name': '调解者',
            'description': '理想主义者，忠于自己的价值观和自己关心的人。希望外部生活与内在价值观一致。好奇心强，很快看到可能性，能够成为实现想法的催化剂。寻求理解人们和帮助他们实现潜力。适应力强，灵活，接受力强，除非危及到价值观。',
            'strengths': ['富有同情心', '理想主义', '创造力', '忠诚', '灵活'],
            'weaknesses': ['过度理想化', '不切实际', '个人情绪化', '自我批评', '对批评敏感']
        },
        'INTP': {
            'name': '逻辑学家',
            'description': '寻求对周围世界原理的逻辑解释。理论导向，对抽象思维比人际互动更感兴趣。安静、内敛、灵活、适应力强。对自己特别感兴趣的领域有非凡的专注能力。',
            'strengths': ['分析能力强', '独立思考', '开放思想', '创新', '好奇心强'],
            'weaknesses': ['可能忽视细节', '缺乏实际性', '健忘', '社交能力有限', '可能过度分析']
        },
        'ESTP': {
            'name': '企业家',
            'description': '善于解决问题，灵活，实用，注重当下，享受周围正在发生的事情。喜欢操作和过程。不喜欢理论和解释。喜欢积极主动地解决问题。',
            'strengths': ['精力充沛', '实用', '灵活', '果断', '解决问题的能力强'],
            'weaknesses': ['不喜欢常规', '可能冒险', '缺乏长期规划', '可能不敏感', '容易分心']
        },
        'ESFP': {
            'name': '表演者',
            'description': '外向、友好、接受力强。热爱生活、人和物质享受。喜欢与他人一起行动。富有常识和实用的能力，使事情变得有趣。善于记住事实而非理论。最擅长处理切实的日常问题和人际关系。',
            'strengths': ['热情', '友好', '现实主义', '实用', '灵活'],
            'weaknesses': ['避免冲突', '容易分心', '冲动', '回避复杂情况', '过于关注社交']
        },
        'ENFP': {
            'name': '活动家',
            'description': '热情洋溢、富有想象力。将生活视为充满可能性。能够很快地将事件和信息联系起来，并自信地根据模式做出决定。迫切需要从他人那里得到肯定，随时准备给予赞赏和支持。灵活、自发，通常依靠口头和语言流利及自信的能力。',
            'strengths': ['热情', '创造力强', '适应力强', '善于沟通', '乐观'],
            'weaknesses': ['难以专注', '计划性差', '情绪化', '过度承诺', '容易感到无聊']
        },
        'ENTP': {
            'name': '辩论家',
            'description': '反应快、足智多谋、善于各种事物。刺激、机警、直言不讳。在解决新的、具有挑战性的问题方面具有资源和才能。善于为构思性问题提出概念模型。技巧娴熟地利用逻辑和口才与他人打交道。',
            'strengths': ['创新', '灵活', '分析能力强', '沟通技巧好', '善于辩论'],
            'weaknesses': ['可能争论性强', '缺乏后续行动', '容易感到无聊', '不重视细节', '优柔寡断']
        },
        'ESTJ': {
            'name': '总经理',
            'description': '实际、现实主义、注重事实，以商业和机械思维为导向。不感兴趣没有实际应用的理论或概念。喜欢组织和管理活动。可能会成为俱乐部、组织和社区事务的管理者。',
            'strengths': ['组织能力强', '务实', '可靠', '决策能力强', '精力充沛'],
            'weaknesses': ['过于直接', '固执', '不灵活', '可能过于支配', '容易判断']
        },
        'ESFJ': {
            'name': '执政官',
            'description': '友好、有良心、合作。希望与他人和谐相处，努力工作以履行义务和职责。忠诚、体贴、关注他人的感受，能够注意到他人需要什么，并努力提供。希望自己和他人的感受被重视和接受。',
            'strengths': ['合作', '关心他人', '可靠', '实用', '有组织'],
            'weaknesses': ['过于顺从', '可能过度保护', '情绪化', '需要认可', '可能过于传统']
        },
        'ENFJ': {
            'name': '主人公',
            'description': '温暖、感情丰富、有责任感、反应灵敏、有社会责任感。非常关注他人的情感、需求和动机。能够看到他人的潜力，并希望帮助他们实现这种潜力。可能在咨询、教导或多样化的艺术领域中服务于他人。',
            'strengths': ['富有同情心', '组织能力强', '忠诚', '善于沟通', '鼓舞人心'],
            'weaknesses': ['可能操控人', '理想主义', '过度自我牺牲', '容易受伤', '固执']
        },
        'ENTJ': {
            'name': '指挥官',
            'description': '坦率、果断，承担领导责任。能够迅速看到程序和政策中的不合逻辑和低效，制定全面的系统来解决组织问题。善于长期规划和目标设定。通常见多识广，博览群书，喜欢扩大知识面并与他人分享他们的见解。',
            'strengths': ['坚定', '战略思维', '领导力强', '决策能力强', '直接坦率'],
            'weaknesses': ['过于强势', '缺乏耐心', '固执', '可能过于批判', '不重视情感因素']
        }
    }
    
    # 维度解释
    dimension_descriptions = {
        'E': '外向型 (Extraversion): 从外部世界获取能量，喜欢广泛的社交互动。',
        'I': '内向型 (Introversion): 从内心世界获取能量，喜欢深度的一对一互动。',
        'S': '感觉型 (Sensing): 关注具体事实和细节，依赖实际经验。',
        'N': '直觉型 (iNtuition): 关注可能性和未来，寻求模式和意义。',
        'T': '思考型 (Thinking): 决策基于逻辑和客观分析。',
        'F': '情感型 (Feeling): 决策基于价值观和对他人的影响。',
        'J': '判断型 (Judging): 喜欢计划、结构和确定性。',
        'P': '感知型 (Perceiving): 喜欢灵活、自发和开放性。'
    }
    
    # 检查MBTI类型是否有效
    if mbti_type not in type_descriptions:
        return {
            'error': f"无效的MBTI类型: {mbti_type}。有效类型包括: {', '.join(type_descriptions.keys())}"
        }
    
    # 获取类型描述
    type_info = type_descriptions[mbti_type]
    
    # 获取维度解释
    dimensions = {
        'attitude': dimension_descriptions[mbti_type[0]],
        'perception': dimension_descriptions[mbti_type[1]],
        'judgment': dimension_descriptions[mbti_type[2]],
        'lifestyle': dimension_descriptions[mbti_type[3]]
    }
    
    # 职业建议
    career_suggestions = {
        'ISTJ': ['会计师', '审计师', '财务分析师', '工程师', '项目管理', '军事或警察工作'],
        'ISFJ': ['护士', '行政助理', '医疗技术人员', '教师', '顾客服务', '社会工作者'],
        'INFJ': ['咨询师', '心理治疗师', '作家/诗人', '教授', '人力资源工作者', '治疗师'],
        'INTJ': ['科学家', '研究员', '系统分析师', '程序员', '工程师', '策略规划师'],
        'ISTP': ['机械师', '工程师', '技术专家', '飞行员', '法医科学家', '建筑师'],
        'ISFP': ['艺术家', '音乐家', '设计师', '厨师', '兽医', '园艺师'],
        'INFP': ['作家', '心理咨询师', '教师', '社会工作者', '图书管理员', '艺术家'],
        'INTP': ['计算机程序员', '科学家', '摄影师', '系统分析师', '大学教授', '工程师'],
        'ESTP': ['企业家', '销售代表', '营销专员', '应急服务人员', '运动员', '娱乐业人士'],
        'ESFP': ['艺人', '销售人员', '教练', '活动策划', '儿童保育', '健康教育者'],
        'ENFP': ['顾问', '记者', '艺术指导', '公共关系专员', '营销人员', '心理咨询师'],
        'ENTP': ['企业家', '律师', '工程师', '创意总监', '发明家', '计算机分析师'],
        'ESTJ': ['经理', '行政人员', '军官', '财务经理', '法官', '项目经理'],
        'ESFJ': ['教师', '销售代表', '医疗保健工作者', '社会工作者', '公共关系专员', '办公室经理'],
        'ENFJ': ['辅导员', '教师', '人力资源专员', '销售培训师', '公共关系专员', '政治家'],
        'ENTJ': ['企业家', '执行官', '律师', '顾问', '业务分析师', '管理顾问']
    }
    
    # 返回完整解释
    return {
        'type': mbti_type,
        'name': type_info['name'],
        'description': type_info['description'],
        'dimensions': dimensions,
        'strengths': type_info['strengths'],
        'weaknesses': type_info['weaknesses'],
        'career_suggestions': career_suggestions[mbti_type]
    }

#################################################
# 主程序
#################################################

def main():
    """主程序入口点"""
    try:
        # 设置命令行参数
        import argparse
        parser = argparse.ArgumentParser(description='MBTI预测系统')
        parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict', 'evaluate'],
                          help='运行模式: train (训练模型), predict (预测MBTI), evaluate (评估模型)')
        parser.add_argument('--file', type=str, default='C:\\Users\\lnasl\\Desktop\\APS360\\APS360\\Data\\Text\\mbti_1.csv',
                          help='数据文件路径 (训练模式)')
        parser.add_argument('--text', type=str, default=None,
                          help='要预测MBTI类型的文本 (预测模式)')
        parser.add_argument('--output', type=str, default=OUTPUT_DIR,
                          help='模型输出目录')
        parser.add_argument('--epochs', type=int, default=8,
                          help='训练轮数')
        parser.add_argument('--batch_size', type=int, default=16,
                          help='批量大小')
        
        args = parser.parse_args()
        
        # 根据模式执行相应操作
        if args.mode == 'train':
            logger.info(f"开始训练模式，使用数据文件: {args.file}")
            results = train_all_models(args.file, args.output, args.epochs, args.batch_size)
            logger.info(f"训练完成，模型保存在: {args.output}")
            
            # 展示一些示例预测
            logger.info("\n=== 示例预测 ===")
            test_texts = [
                "I enjoy spending time alone reading books and thinking about deep philosophical questions.",
                "I love organizing events and bringing people together. Structure and planning are important to me.",
                "I make decisions based on logic and facts rather than emotions or how others might feel."
            ]
            
            for i, text in enumerate(test_texts):
                prediction = predict_mbti_type(text, model_dir=args.output)
                explanation = explain_mbti_type(prediction['ensemble_type'])
                
                logger.info(f"\n样本 {i+1}: {text[:50]}...")
                logger.info(f"预测MBTI类型: {prediction['ensemble_type']} ({explanation['name']}) - 置信度: {prediction['confidence']:.4f}")
                logger.info(f"描述: {explanation['description']}")
                logger.info(f"优势: {', '.join(explanation['strengths'])}")
                logger.info(f"劣势: {', '.join(explanation['weaknesses'])}")
                logger.info(f"适合职业: {', '.join(explanation['career_suggestions'][:3])}")
            
        elif args.mode == 'predict':
            if args.text is None:
                # 交互模式
                logger.info("MBTI预测系统 - 交互模式")
                logger.info("输入文本以预测MBTI类型，输入'exit'退出")
                
                # 加载模型（只加载一次）
                models = load_ensemble_model(args.output)
                
                while True:
                    text = input("\n请输入文本（或输入'exit'退出）: ")
                    if text.lower() == 'exit':
                        break
                    
                    prediction = predict_mbti_type(text, models, args.output)
                    explanation = explain_mbti_type(prediction['ensemble_type'])
                    
                    print(f"\n预测MBTI类型: {prediction['ensemble_type']} ({explanation['name']})")
                    print(f"置信度: {prediction['confidence']:.4f}")
                    print(f"\n描述: {explanation['description']}")
                    print(f"\n优势: {', '.join(explanation['strengths'])}")
                    print(f"\n劣势: {', '.join(explanation['weaknesses'])}")
                    print(f"\n适合职业: {', '.join(explanation['career_suggestions'])}")
                    
                    # 显示模型预测详情
                    print("\n各模型预测结果:")
                    for model, pred in prediction['model_predictions'].items():
                        print(f"  {model}: {pred}")
                    
                    # 显示备选类型
                    if prediction['alternative_types']:
                        print("\n其他可能的类型:")
                        for type_name, prob in prediction['alternative_types']:
                            alt_explanation = explain_mbti_type(type_name)
                            print(f"  {type_name} ({alt_explanation['name']}): {prob:.4f}")
            else:
                # 单次预测模式
                prediction = predict_mbti_type(args.text, model_dir=args.output)
                explanation = explain_mbti_type(prediction['ensemble_type'])
                
                print(f"\n文本: {args.text}")
                print(f"\n预测MBTI类型: {prediction['ensemble_type']} ({explanation['name']})")
                print(f"置信度: {prediction['confidence']:.4f}")
                print(f"\n描述: {explanation['description']}")
                print(f"\n优势: {', '.join(explanation['strengths'])}")
                print(f"\n劣势: {', '.join(explanation['weaknesses'])}")
                print(f"\n适合职业: {', '.join(explanation['career_suggestions'])}")
        
        elif args.mode == 'evaluate':
            logger.info(f"开始评估模式，使用模型目录: {args.output}")
            
            # 加载模型
            models = load_ensemble_model(args.output)
            
            # 加载测试数据
            processor = models['processor']
            data = load_and_process_data(args.file, processor)
            
            # 评估模型
            ensemble_correct = 0
            ensemble_total = 0
            
            for i in range(len(data['X_test'])):
                # 获取样本文本
                text = data['X_test']['cleaned_posts'].iloc[i]
                
                # 进行集成预测
                prediction = predict_mbti_type(text, models, args.output)
                
                # 获取真实标签
                true_type = data['label_encoder'].inverse_transform([data['y_test']['encoded_type'].iloc[i]])[0]
                
                # 统计准确率
                if prediction['ensemble_type'] == true_type:
                    ensemble_correct += 1
                
                ensemble_total += 1
            
            # 计算集成准确率
            ensemble_accuracy = ensemble_correct / ensemble_total
            logger.info(f"集成模型准确率: {ensemble_accuracy:.4f} ({ensemble_correct}/{ensemble_total})")
    
    except Exception as e:
        logger.error(f"运行时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    # 确保导入F
    import torch.nn.functional as F
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    main()