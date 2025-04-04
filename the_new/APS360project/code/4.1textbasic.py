# -*- coding: utf-8 -*-
"""
MBTI 性格类型预测模型 (无 WordNet 依赖)
此版本解决了 zero_division 警告并增强了对稀有类型的处理能力
"""

import os
import re
import string
import pandas as pd
import numpy as np
import pickle
import json
import time
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline

# 全局配置
MODEL_DIR = r"C:\Users\lnasl\Desktop\APS360\APS360\Model"
# 新的最佳模型保存路径
BEST_MODEL_DIR = r"C:\Users\lnasl\Desktop\APS360\APS360\Model\TrainedModel\text\ml"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)  # 确保目标目录存在

# 1. 强大的文本预处理函数 (不依赖 NLTK WordNet)
def robust_text_preprocessing(text):
    """
    对文本进行全面预处理，不依赖 NLTK WordNet
    
    参数:
        text (str): 待处理的原始文本
        
    返回:
        str: 处理后的文本
    """
    # 确保文本是字符串
    text = str(text).lower()
    
    # 删除 URL
    text = re.sub(r'https?://\S+|www\.\S+', ' url ', text)
    
    # 删除 HTML 标签
    text = re.sub(r'<.*?>', ' ', text)
    
    # 删除电子邮件地址
    text = re.sub(r'\S+@\S+', ' email ', text)
    
    # 将数字替换为 'number' 标记
    text = re.sub(r'\d+', ' number ', text)
    
    # 处理标点符号
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # 自定义简单的词干提取
    words = text.split()
    processed_words = []
    
    # 英语停用词列表（常用）
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
            
        # 跳过极短的词
        if len(word) <= 2:
            continue
            
        # 简单的词干提取规则
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
    
    # 删除多余的空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# 2. 数据加载和分析
def load_data(file_path):
    """
    加载 MBTI 数据集，并确保格式正确
    
    参数:
        file_path (str): CSV 或 Excel 文件路径
        
    返回:
        DataFrame: 加载的数据
    """
    print(f"加载数据: {file_path}")
    
    try:
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='latin1')
        except Exception as e:
            print(f"尝试其他编码: {e}")
            encodings = ['cp1252', 'ISO-8859-1', 'cp850']
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    print(f"成功使用 {encoding} 编码")
                    break
                except:
                    continue
    
    # 确保列名正确
    if 'type' not in df.columns or 'posts' not in df.columns:
        # 尝试识别包含 MBTI 类型和文本的列
        type_col = None
        post_col = None
        
        # 寻找包含 MBTI 类型的列
        for col in df.columns:
            sample_values = df[col].astype(str).str.upper().tolist()[:20]
            mbti_count = sum(1 for val in sample_values if re.match(r'^[IE][NS][TF][JP]$', val.strip()))
            if mbti_count > len(sample_values) * 0.5:
                type_col = col
                print(f"找到 MBTI 类型列: {col}")
                break
        
        # 寻找包含文本数据的列（最长文本）
        text_lens = {col: df[col].astype(str).str.len().mean() for col in df.columns if col != type_col}
        if text_lens:
            post_col = max(text_lens, key=text_lens.get)
            print(f"找到文本数据列: {post_col}")
            
        if type_col and post_col:
            df = df[[type_col, post_col]]
            df.columns = ['type', 'posts']
        else:
            raise ValueError("无法识别 MBTI 类型和帖子列")
    
    # 数据概览
    print(f"数据集大小: {df.shape}")
    print(f"空值统计:\n{df.isnull().sum()}")
    
    # 替换 NaN 值
    df = df.fillna({'type': '', 'posts': ''})
    
    return df

def analyze_mbti_data(df):
    """
    深入分析 MBTI 数据集的特征
    
    参数:
        df (DataFrame): 原始数据
        
    返回:
        DataFrame: 有效的数据（过滤掉无效的 MBTI 类型）
    """
    # 确保类型列大写并去除空格
    df['type'] = df['type'].str.upper().str.strip()
    
    # 过滤无效的 MBTI 类型
    valid_df = df[df['type'].str.match(r'^[IE][NS][TF][JP]$')]
    invalid_count = len(df) - len(valid_df)
    print(f"有效的 MBTI 类型: {len(valid_df)} / {len(df)} (移除了 {invalid_count} 行无效数据)")
    
    # 探索各类型的分布情况
    type_counts = Counter(valid_df['type'])
    total = len(valid_df)
    
    print("\nMBTI 类型分布:")
    for mbti_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        percentage = (count / total) * 100
        print(f"{mbti_type}: {count} ({percentage:.2f}%)")
    
    # 深入分析四个维度
    dimensions = {
        'I/E': {'I': 0, 'E': 0},
        'N/S': {'N': 0, 'S': 0},
        'T/F': {'T': 0, 'F': 0},
        'J/P': {'J': 0, 'P': 0}
    }
    
    for mbti_type in valid_df['type']:
        dimensions['I/E'][mbti_type[0]] = dimensions['I/E'].get(mbti_type[0], 0) + 1
        dimensions['N/S'][mbti_type[1]] = dimensions['N/S'].get(mbti_type[1], 0) + 1
        dimensions['T/F'][mbti_type[2]] = dimensions['T/F'].get(mbti_type[2], 0) + 1
        dimensions['J/P'][mbti_type[3]] = dimensions['J/P'].get(mbti_type[3], 0) + 1
    
    print("\n各维度分布:")
    for dimension, counts in dimensions.items():
        total_dim = sum(counts.values())
        for trait, count in counts.items():
            percentage = (count / total_dim) * 100
            print(f"{dimension} - {trait}: {count} ({percentage:.2f}%)")
    
    # 文本长度分析
    valid_df['text_length'] = valid_df['posts'].astype(str).str.len()
    avg_length = valid_df['text_length'].mean()
    max_length = valid_df['text_length'].max()
    min_length = valid_df['text_length'].min()
    
    print(f"\n文本长度统计:")
    print(f"平均长度: {avg_length:.2f} 字符")
    print(f"最大长度: {max_length} 字符")
    print(f"最小长度: {min_length} 字符")
    
    return valid_df

# 3. 高级数据增强和平衡
def enhance_class_balance(df, target_ratio=0.7):
    """
    增强数据平衡性，为稀有类型创建合成样本
    
    参数:
        df (DataFrame): 原始数据集
        target_ratio (float): 目标样本比例（相对于最大类别的数量）
        
    返回:
        DataFrame: 平衡后的数据集
    """
    print("增强数据集平衡性...")
    
    # 计算各类型的样本数
    type_counts = Counter(df['type'])
    max_count = max(type_counts.values())
    target_count = int(max_count * target_ratio)  # 设定目标样本数
    
    print(f"最大类别样本数: {max_count}")
    print(f"目标样本数: {target_count}")
    
    balanced_data = []
    
    # 添加原始数据
    balanced_data.append(df)
    
    # 为每个类别增强数据
    for mbti_type, count in type_counts.items():
        # 只对样本数少于目标值的类型进行增强
        if count < target_count:
            print(f"增强类型 {mbti_type}: 当前 {count} -> 目标 {target_count}")
            type_data = df[df['type'] == mbti_type]
            
            # 计算需要额外创建的样本数
            samples_needed = target_count - count
            
            # 增强数据
            enhanced_samples = []
            
            # 创建所需数量的合成样本
            for _ in range(samples_needed):
                # 随机选择类型中的一个样本
                sample = type_data.sample(1).iloc[0]
                
                # 创建增强版本
                processed_text = sample['posts']
                words = processed_text.split()
                
                if len(words) >= 10:
                    # 随机打乱部分词序
                    split_point = len(words) // 3
                    if split_point > 0:
                        first_part = words[:split_point]
                        np.random.shuffle(first_part)
                        words = first_part + words[split_point:]
                    
                    # 在随机位置插入类型特定的关键词
                    mbti_keywords = get_mbti_keywords(mbti_type)
                    selected_keywords = np.random.choice(
                        mbti_keywords, 
                        size=min(3, len(mbti_keywords)), 
                        replace=False
                    )
                    
                    # 随机插入位置
                    for keyword in selected_keywords:
                        insert_pos = np.random.randint(0, len(words))
                        words.insert(insert_pos, keyword)
                
                # 创建增强文本
                enhanced_text = ' '.join(words)
                
                # 创建增强样本
                enhanced_sample = sample.copy()
                enhanced_sample['posts'] = enhanced_text
                enhanced_samples.append(enhanced_sample)
            
            # 添加增强样本到平衡数据集
            if enhanced_samples:
                balanced_data.append(pd.DataFrame(enhanced_samples))
    
    # 合并所有数据
    balanced_df = pd.concat(balanced_data, ignore_index=True)
    print(f"数据平衡后大小: {balanced_df.shape}")
    
    return balanced_df

def get_mbti_keywords(mbti_type):
    """
    获取与特定 MBTI 类型相关的关键词
    
    参数:
        mbti_type (str): MBTI 类型 (例如 "INTJ")
        
    返回:
        list: 关键词列表
    """
    # 各维度的关键词
    dimension_keywords = {
        'I': ['introvert', 'quiet', 'reflect', 'alone', 'private', 'inner', 'depth', 'focus', 'thought'],
        'E': ['extrovert', 'social', 'talk', 'people', 'engage', 'outgoing', 'active', 'external', 'interact'],
        'N': ['intuitive', 'abstract', 'future', 'imagine', 'possibility', 'pattern', 'meaning', 'theory', 'concept'],
        'S': ['sensing', 'detail', 'present', 'practical', 'concrete', 'reality', 'fact', 'experience', 'observation'],
        'T': ['thinking', 'logic', 'analysis', 'objective', 'principle', 'rational', 'critique', 'reason', 'system'],
        'F': ['feeling', 'value', 'harmony', 'empathy', 'personal', 'compassion', 'ethic', 'human', 'subjective'],
        'J': ['judging', 'plan', 'organize', 'structure', 'decide', 'control', 'certain', 'schedule', 'complete'],
        'P': ['perceiving', 'flexible', 'adapt', 'explore', 'option', 'spontaneous', 'open', 'process', 'possibility']
    }
    
    # 为每个 MBTI 类型的每个字母添加关键词
    keywords = []
    for letter in mbti_type:
        if letter in dimension_keywords:
            keywords.extend(dimension_keywords[letter])
    
    return keywords

# 4. 特征工程和模型准备
def prepare_features(df):
    """
    准备用于训练的特征
    
    参数:
        df (DataFrame): 预处理后的数据集
        
    返回:
        tuple: 处理后的特征、标签、向量化器和标签编码器
    """
    print("准备特征...")
    
    # 预处理文本
    df['processed_posts'] = df['posts'].apply(robust_text_preprocessing)
    
    # 准备标签
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['type'])
    
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_posts'], y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    # 创建 TF-IDF 向量化器
    vectorizer = TfidfVectorizer(
        max_features=15000,  # 增加特征数量
        ngram_range=(1, 3),  # 1-3 gram 捕捉短语
        min_df=2,            # 至少出现在 2 个文档中
        max_df=0.9,          # 忽略出现在超过 90% 文档中的词
        sublinear_tf=True,   # 次线性 TF 缩放
        use_idf=True,        # 使用 IDF
        stop_words='english' # 使用内置的英语停用词
    )
    
    # 转换文本数据
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"训练集特征形状: {X_train_tfidf.shape}")
    print(f"测试集特征形状: {X_test_tfidf.shape}")
    
    # 计算类别权重（解决类别不平衡）
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    n_classes = len(class_counts)
    
    class_weights = {i: total_samples / (n_classes * count) for i, count in enumerate(class_counts)}
    print("类别权重:")
    for i, weight in class_weights.items():
        mbti_type = label_encoder.inverse_transform([i])[0]
        print(f"  {mbti_type}: {weight:.2f}")
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer, label_encoder, class_weights

# 5. 模型训练和评估
def train_and_evaluate_models(X_train, X_test, y_train, y_test, vectorizer, label_encoder, class_weights):
    """
    训练和评估多个模型
    
    参数:
        X_train, X_test: 训练和测试特征
        y_train, y_test: 训练和测试标签
        vectorizer: TF-IDF 向量化器
        label_encoder: 标签编码器
        class_weights: 类别权重
        
    返回:
        tuple: 模型结果和最佳模型名称
    """
    models = {
        'LogisticRegression': LogisticRegression(
            max_iter=2000,           # 增加迭代次数
            C=1.0,                   # 正则化强度
            class_weight=class_weights,  # 使用计算的类别权重
            solver='liblinear',      # 适用于小数据集
            multi_class='ovr',       # 一对多策略
            random_state=42
        ),
        'LinearSVC': LinearSVC(
            max_iter=2000,          
            C=1.0,                  
            class_weight=class_weights,
            dual=False,              # 对于大特征集更快
            random_state=42
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=200,        # 增加树的数量
            max_depth=40,            # 增加树的深度
            min_samples_split=5,     # 分裂所需的最小样本数
            min_samples_leaf=2,      # 叶节点所需的最小样本数
            class_weight=class_weights,
            random_state=42,
            n_jobs=-1                # 使用所有可用的处理器
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n训练 {name} 模型...")
        
        # 训练模型
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # 预测和评估
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"{name} 模型训练时间: {train_time:.2f} 秒")
        print(f"{name} 模型准确率: {accuracy:.4f}")
        print(f"{name} 模型 F1 分数: {f1:.4f}")
        
        # 分类报告（处理零除警告）
        target_names = label_encoder.classes_
        report = classification_report(
            y_test, y_pred,
            target_names=target_names,
            digits=4,
            zero_division=0  # 设置零除行为，避免警告
        )
        print(f"分类报告:\n{report}")
        
        # 保存结果
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'f1_score': f1,
            'train_time': train_time,
            'report': classification_report(
                y_test, y_pred,
                target_names=target_names,
                digits=4,
                zero_division=0,
                output_dict=True
            )
        }
    
    # 找到最佳模型
    best_model_name = max(results, key=lambda k: results[k]['accuracy'])
    print(f"\n最佳模型: {best_model_name} (准确率: {results[best_model_name]['accuracy']:.4f})")
    
    return results, best_model_name

# 6. 分析模型预测
def analyze_predictions(model, X_test, y_test, label_encoder):
    """
    详细分析模型预测结果
    
    参数:
        model: 训练好的模型
        X_test: 测试特征
        y_test: 测试标签
        label_encoder: 标签编码器
        
    返回:
        dict: 分析结果
    """
    print("\n分析模型预测...")
    
    # 获取预测
    y_pred = model.predict(X_test)
    
    # 创建混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    class_names = label_encoder.classes_
    
    # 分析每个类别的性能
    class_performance = {}
    for i, class_name in enumerate(class_names):
        # 真正例、真负例、假正例、假负例
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        
        # 计算指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_performance[class_name] = {
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_negatives': int(tn),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'support': int(cm[i, :].sum())
        }
    
    # 打印性能最差的类别
    worst_classes = sorted(class_performance.items(), key=lambda x: x[1]['f1_score'])[:3]
    print("性能最差的 MBTI 类型:")
    for class_name, metrics in worst_classes:
        print(f"  {class_name}: F1={metrics['f1_score']:.4f}, 精确率={metrics['precision']:.4f}, 召回率={metrics['recall']:.4f}")
    
    return {
        'confusion_matrix': cm.tolist(),
        'class_performance': class_performance
    }

# 7. 保存和加载模型
def save_model(model, vectorizer, label_encoder, model_name, metadata=None):
    """
    保存模型及相关组件
    
    参数:
        model: 训练好的模型
        vectorizer: TF-IDF 向量化器
        label_encoder: 标签编码器
        model_name: 模型名称
        metadata: 额外元数据
        
    返回:
        str: 模型保存路径
    """
    model_path = os.path.join(MODEL_DIR, model_name)
    os.makedirs(model_path, exist_ok=True)
    
    # 保存模型
    with open(os.path.join(model_path, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    # 保存向量化器
    with open(os.path.join(model_path, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # 保存标签编码器
    with open(os.path.join(model_path, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # 创建配置
    config = {
        'model_name': model_name,
        'classes': label_encoder.classes_.tolist(),
        'feature_count': vectorizer.max_features,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 添加额外元数据
    if metadata:
        config.update(metadata)
    
    # 保存配置
    with open(os.path.join(model_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"模型已保存到: {model_path}")
    
    return model_path

def save_best_model(results, best_model_name, vectorizer, label_encoder, analysis=None):
    """
    将最佳模型保存到专用目录
    
    参数:
        results: 训练结果
        best_model_name: 最佳模型名称
        vectorizer: TF-IDF 向量化器
        label_encoder: 标签编码器
        analysis: 模型分析结果
        
    返回:
        str: 最佳模型保存路径
    """
    best_model = results[best_model_name]['model']
    
    # 准备元数据
    metadata = {
        'accuracy': results[best_model_name]['accuracy'],
        'f1_score': results[best_model_name]['f1_score'],
        'train_time': results[best_model_name]['train_time'],
        'report': results[best_model_name]['report']
    }
    
    if analysis:
        metadata['analysis'] = analysis
    
    # 保存到新的指定目录
    best_model_path = BEST_MODEL_DIR
    if os.path.exists(best_model_path):
        import shutil
        shutil.rmtree(best_model_path)
    
    os.makedirs(best_model_path, exist_ok=True)
    
    # 保存模型
    with open(os.path.join(best_model_path, 'model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
    
    # 保存向量化器
    with open(os.path.join(best_model_path, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # 保存标签编码器
    with open(os.path.join(best_model_path, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # 保存配置
    config = {
        'model_name': best_model_name,
        'classes': label_encoder.classes_.tolist(),
        'feature_count': vectorizer.max_features,
        'accuracy': results[best_model_name]['accuracy'],
        'f1_score': results[best_model_name]['f1_score'],
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(best_model_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # 保存详细报告
    with open(os.path.join(best_model_path, 'report.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"最佳模型已保存到: {best_model_path}")
    
    return best_model_path

def load_best_model():
    """
    加载最佳模型
    
    返回:
        tuple: 模型、向量化器、标签编码器和配置
    """
    # 从新的位置加载最佳模型
    best_model_path = BEST_MODEL_DIR
    
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"未找到最佳模型目录: {best_model_path}")
    
    # 加载模型
    with open(os.path.join(best_model_path, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    
    # 加载向量化器
    with open(os.path.join(best_model_path, 'vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
    
    # 加载标签编码器
    with open(os.path.join(best_model_path, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)
    
    # 加载配置
    with open(os.path.join(best_model_path, 'config.json'), 'r') as f:
        config = json.load(f)
    
    return model, vectorizer, label_encoder, config

# 8. 高级预测功能
def predict_mbti_type(text, model=None, vectorizer=None, label_encoder=None):
    """
    预测文本的 MBTI 类型，支持段落输入
    
    参数:
        text (str): 输入文本
        model, vectorizer, label_encoder: 可选组件（如未提供则加载最佳模型）
        
    返回:
        dict: 预测结果
    """
    # 加载模型组件（如果未提供）
    if model is None or vectorizer is None or label_encoder is None:
        try:
            model, vectorizer, label_encoder, _ = load_best_model()
        except Exception as e:
            print(f"加载模型失败: {e}")
            return None
    
    # 预处理输入文本
    processed_text = robust_text_preprocessing(text)
    
    # 向量化文本
    text_vector = vectorizer.transform([processed_text])
    
    # 使用标准模型预测
    if hasattr(model, 'predict_proba'):
        # 获取预测概率
        proba = model.predict_proba(text_vector)[0]
        prediction = proba.argmax()
        confidence = proba[prediction]
    else:
        # 不支持概率的模型
        prediction = model.predict(text_vector)[0]
        confidence = 1.0  # 默认置信度
    
    mbti_type = label_encoder.inverse_transform([prediction])[0]
    
    # 增强型预测：分析每个 MBTI 维度
    ie_score, ns_score, tf_score, jp_score = analyze_mbti_dimensions(text, processed_text)
    
    # 基于维度分析的分数确定类型（备选预测）
    alt_type = ''
    alt_type += 'I' if ie_score < 0 else 'E'
    alt_type += 'N' if ns_score > 0 else 'S'
    alt_type += 'T' if tf_score < 0 else 'F'
    alt_type += 'J' if jp_score < 0 else 'P'
    
    # 构建返回结果
    result = {
        'mbti_type': mbti_type,
        'confidence': float(confidence),
        'alternative_type': alt_type,
        'dimension_analysis': {
            'IE': {'score': ie_score, 'preference': 'I' if ie_score < 0 else 'E'},
            'NS': {'score': ns_score, 'preference': 'N' if ns_score > 0 else 'S'},
            'TF': {'score': tf_score, 'preference': 'T' if tf_score < 0 else 'F'},
            'JP': {'score': jp_score, 'preference': 'J' if jp_score < 0 else 'P'}
        }
    }
    
    # 如果支持概率输出，添加前三个最可能的类型
    if hasattr(model, 'predict_proba'):
        top_indices = proba.argsort()[-3:][::-1]
        result['top_predictions'] = [
            {
                'type': label_encoder.inverse_transform([idx])[0],
                'confidence': float(proba[idx])
            } for idx in top_indices
        ]
    
    return result

def analyze_mbti_dimensions(text, processed_text=None):
    """
    分析文本在四个 MBTI 维度上的得分
    
    参数:
        text (str): 原始文本
        processed_text (str): 已预处理的文本（可选）
        
    返回:
        tuple: 四个维度的得分 (IE, NS, TF, JP)
    """
    if processed_text is None:
        processed_text = robust_text_preprocessing(text)
    
    # 每个维度的关键词和它们的权重
    dimension_keywords = {
        'I': {'introvert': 2, 'quiet': 1, 'reflect': 1, 'alone': 1, 'private': 1, 'inner': 1, 
              'depth': 1, 'focus': 1, 'thought': 1, 'peace': 1, 'solitude': 2, 'individual': 1},
        'E': {'extrovert': 2, 'social': 1, 'talk': 1, 'people': 1, 'engage': 1, 'outgoing': 1, 
              'active': 1, 'external': 1, 'interact': 1, 'energetic': 1, 'group': 1, 'party': 1},
        'N': {'intuitive': 2, 'abstract': 1, 'future': 1, 'imagine': 1, 'possibility': 1, 'pattern': 1, 
              'meaning': 1, 'theory': 1, 'concept': 1, 'insight': 1, 'innovative': 1, 'vision': 1},
        'S': {'sensing': 2, 'detail': 1, 'present': 1, 'practical': 1, 'concrete': 1, 'reality': 1, 
              'fact': 1, 'experience': 1, 'observation': 1, 'specific': 1, 'actual': 1, 'tangible': 1},
        'T': {'thinking': 2, 'logic': 1, 'analysis': 1, 'objective': 1, 'principle': 1, 'rational': 1, 
              'critique': 1, 'reason': 1, 'system': 1, 'truth': 1, 'fair': 1, 'consistent': 1},
        'F': {'feeling': 2, 'value': 1, 'harmony': 1, 'empathy': 1, 'personal': 1, 'compassion': 1, 
              'ethic': 1, 'human': 1, 'subjective': 1, 'emotion': 1, 'care': 1, 'moral': 1},
        'J': {'judging': 2, 'plan': 1, 'organize': 1, 'structure': 1, 'decide': 1, 'control': 1, 
              'certain': 1, 'schedule': 1, 'complete': 1, 'deadline': 1, 'goal': 1, 'closure': 1},
        'P': {'perceiving': 2, 'flexible': 1, 'adapt': 1, 'explore': 1, 'option': 1, 'spontaneous': 1, 
              'open': 1, 'process': 1, 'possibility': 1, 'casual': 1, 'flow': 1, 'discover': 1}
    }
    
    # 计算每个维度的得分
    ie_score = 0  # 负值偏向 I，正值偏向 E
    ns_score = 0  # 正值偏向 N，负值偏向 S
    tf_score = 0  # 负值偏向 T，正值偏向 F
    jp_score = 0  # 负值偏向 J，正值偏向 P
    
    # 处理文本词语
    words = processed_text.split()
    
    # 计算维度得分
    for word in words:
        # I/E 维度
        for keyword, weight in dimension_keywords['I'].items():
            if keyword in word:
                ie_score -= weight
        for keyword, weight in dimension_keywords['E'].items():
            if keyword in word:
                ie_score += weight
        
        # N/S 维度
        for keyword, weight in dimension_keywords['N'].items():
            if keyword in word:
                ns_score += weight
        for keyword, weight in dimension_keywords['S'].items():
            if keyword in word:
                ns_score -= weight
        
        # T/F 维度
        for keyword, weight in dimension_keywords['T'].items():
            if keyword in word:
                tf_score -= weight
        for keyword, weight in dimension_keywords['F'].items():
            if keyword in word:
                tf_score += weight
        
        # J/P 维度
        for keyword, weight in dimension_keywords['J'].items():
            if keyword in word:
                jp_score -= weight
        for keyword, weight in dimension_keywords['P'].items():
            if keyword in word:
                jp_score += weight
    
    # 归一化得分
    def normalize_score(score, words_count):
        if words_count == 0:
            return 0
        normalized = score / (words_count ** 0.5)  # 使用平方根缩放，减少长文本的优势
        return max(min(normalized, 10), -10)  # 限制在 -10 到 10 的范围内
    
    words_count = len(words)
    ie_score = normalize_score(ie_score, words_count)
    ns_score = normalize_score(ns_score, words_count)
    tf_score = normalize_score(tf_score, words_count)
    jp_score = normalize_score(jp_score, words_count)
    
    return ie_score, ns_score, tf_score, jp_score

# 9. 综合训练流程
def train_mbti_prediction_model(file_path):
    """
    训练 MBTI 预测模型的主函数
    
    参数:
        file_path (str): 数据文件路径
        
    返回:
        dict: 训练结果
    """
    print("="*80)
    print("开始训练 MBTI 性格类型预测模型")
    print("="*80)
    
    # 1. 加载和分析数据
    raw_df = load_data(file_path)
    valid_df = analyze_mbti_data(raw_df)
    
    # 2. 增强类别平衡
    balanced_df = enhance_class_balance(valid_df)
    
    # 3. 准备特征
    X_train, X_test, y_train, y_test, vectorizer, label_encoder, class_weights = prepare_features(balanced_df)
    
    # 4. 训练和评估模型
    results, best_model_name = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, vectorizer, label_encoder, class_weights
    )
    
    # 5. 详细分析最佳模型
    best_model = results[best_model_name]['model']
    analysis = analyze_predictions(best_model, X_test, y_test, label_encoder)
    
    # 6. 保存所有模型
    for name, result in results.items():
        model_metadata = {
            'accuracy': result['accuracy'],
            'f1_score': result['f1_score'],
            'train_time': result['train_time'],
            'report': result['report']
        }
        save_model(result['model'], vectorizer, label_encoder, name, model_metadata)
    
    # 7. 保存最佳模型
    best_model_path = save_best_model(results, best_model_name, vectorizer, label_encoder, analysis)
    
    print("\n模型训练完成!")
    print(f"最佳模型 ({best_model_name}) 已保存到: {best_model_path}")
    
    return {
        'best_model': best_model_name,
        'accuracy': results[best_model_name]['accuracy'],
        'f1_score': results[best_model_name]['f1_score'],
        'best_model_path': best_model_path
    }

# 10. 测试和演示
def test_model_with_examples():
    """
    使用示例文本测试加载的模型
    """
    print("\n测试模型预测...")
    
    # 加载最佳模型
    try:
        model, vectorizer, label_encoder, config = load_best_model()
        print(f"成功加载模型。模型类型: {config['model_name']}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 测试文本集合
    test_texts = [
        """I like spending time alone, reflecting on the universe and the meaning of life. 
        I'm always looking for deeper understanding and connections between seemingly 
        unrelated concepts. I enjoy theoretical discussions more than practical matters.""",
        
        """I love being around people and organizing social events. I'm very practical
        and focus on immediate results. I believe in traditions and value stability.
        I prefer clear rules and established procedures.""",
        
        """I make decisions based on logic and objective analysis. I enjoy solving 
        complex problems and finding efficient solutions. I value competence and
        intellectual discussions. I tend to focus on concepts rather than details."""
    ]
    
    # 预测每个测试文本
    for i, text in enumerate(test_texts):
        result = predict_mbti_type(text, model, vectorizer, label_encoder)
        
        print(f"\n测试文本 {i+1}:")
        print(f"预测的 MBTI 类型: {result['mbti_type']} (置信度: {result['confidence']*100:.2f}%)")
        print(f"备选类型: {result['alternative_type']}")
        
        print("维度分析:")
        for dim, analysis in result['dimension_analysis'].items():
            print(f"  {dim}: {analysis['score']:.2f} (偏向 {analysis['preference']})")
        
        if 'top_predictions' in result:
            print("前三个最可能的类型:")
            for pred in result['top_predictions']:
                print(f"  {pred['type']}: {pred['confidence']*100:.2f}%")
    
    print("\n测试完成")

# 主函数
if __name__ == "__main__":
    # 数据文件路径
    file_path = r"C:\Users\lnasl\Desktop\APS360\APS360\Data\Text\mbti_1.csv"
    
    # 选择操作模式
    print("请选择操作：")
    print("1. 训练新模型")
    print("2. 测试现有模型")
    
    choice = input("输入选项 (1/2): ")
    
    if choice == "1":
        # 训练模型
        results = train_mbti_prediction_model(file_path)
        print(f"模型训练结果: {results}")
        
        # 测试训练好的模型
        test_model_with_examples()
    elif choice == "2":
        # 仅测试现有模型
        test_model_with_examples()
    else:
        print("无效选项")