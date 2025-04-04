import pandas as pd
import numpy as np
import re
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
import random
import matplotlib.pyplot as plt
import matplotlib

# 设置字体为 SimHei（黑体），支持中文
matplotlib.rcParams['font.family'] = 'SimHei'

# 避免负号显示为方块的问题
matplotlib.rcParams['axes.unicode_minus'] = False


# 下载NLTK资源（如果尚未下载）
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 路径设置
DATA_PATH = r"C:\Users\lnasl\Desktop\APS360\APS360\Data\Text\mbti_1.csv"
MODEL_SAVE_PATH = r"C:\Users\lnasl\Desktop\APS360\APS360\Model\TrainedModel\text\improved_dl"
GLOVE_PATH = r"glove.6B.100d.txt"  # 如果有GloVe文件，请指定正确路径

# 创建目录（如果不存在）
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# 超参数设置
MAX_VOCAB_SIZE = 20000
MAX_SEQ_LENGTH = 500
EMBEDDING_DIM = 100  # 更改为与GloVe向量维度匹配
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
NUM_EPOCHS = 50  # 增加训练轮数
CLIP = 5  # 梯度裁剪
EARLY_STOP_PATIENCE = 5  # 早停参数

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 数据预处理函数
def clean_text(text):
    """清洗和预处理文本数据"""
    # 转换为小写
    text = text.lower()
    
    # 移除URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 移除HTML标签
    text = re.sub(r'<.*?>', '', text)
    
    # 移除标点符号
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_text(text, stop_words=None):
    """分词并可选地移除停用词"""
    tokens = word_tokenize(text)
    # 注意：我们保留停用词，因为它们可能包含个性信号
    # if stop_words:
    #     tokens = [word for word in tokens if word not in stop_words]
    return tokens

# 数据增强函数
def get_synonyms(word):
    """获取单词的同义词"""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym != word:
                synonyms.add(synonym)
    return list(synonyms)

def synonym_replacement(tokens, n=2):
    """通过同义词替换来增强文本"""
    if not tokens:
        return tokens
    
    stop_words = set(stopwords.words('english'))
    new_tokens = tokens.copy()
    # 获取非停用词列表
    non_stop_words = [word for word in tokens if word not in stop_words]
    
    # 如果没有非停用词，返回原始tokens
    if not non_stop_words:
        return tokens
    
    # 随机打乱非停用词列表
    random.shuffle(non_stop_words)
    num_replaced = 0
    
    for word in non_stop_words:
        synonyms = get_synonyms(word)
        if synonyms:
            # 选择一个随机同义词
            synonym = random.choice(synonyms)
            # 替换词
            new_tokens = [synonym if token == word else token for token in new_tokens]
            num_replaced += 1
        
        # 达到指定替换次数后停止
        if num_replaced >= n:
            break
            
    return new_tokens

# 加载和预处理数据
print("加载和预处理数据...")
df = pd.read_csv(DATA_PATH)

# 显示数据集信息
print(f"数据集形状: {df.shape}")
print(f"MBTI类型分布:\n{df['type'].value_counts()}")

# 清洗并分词帖子
stop_words = set(stopwords.words('english'))
df['cleaned_posts'] = df['posts'].apply(clean_text)
df['tokenized_posts'] = df['cleaned_posts'].apply(lambda x: tokenize_text(x))

# 对训练数据进行增强
print("对数据进行增强...")
# 我们先创建训练集和验证集的分割
X_temp = df['tokenized_posts'].tolist()
y_temp = df['type'].tolist()
X_train_temp, X_val, y_train_temp, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
)

# 对训练数据进行增强
X_train_augmented = []
y_train_augmented = []

for tokens, label in zip(X_train_temp, y_train_temp):
    # 添加原始样本
    X_train_augmented.append(tokens)
    y_train_augmented.append(label)
    
    # 添加增强样本（同义词替换）
    for _ in range(1):  # 每个原始样本添加1个增强样本
        augmented_tokens = synonym_replacement(tokens)
        X_train_augmented.append(augmented_tokens)
        y_train_augmented.append(label)

print(f"增强前训练样本数: {len(X_train_temp)}")
print(f"增强后训练样本数: {len(X_train_augmented)}")

# 构建词汇表
print("构建词汇表...")
word_counts = Counter()
for tokens in X_train_augmented:
    word_counts.update(tokens)

# 只保留最常见的MAX_VOCAB_SIZE个词
vocab = ["<PAD>", "<UNK>"] + [word for word, _ in word_counts.most_common(MAX_VOCAB_SIZE - 2)]
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

# 编码MBTI类型
label_encoder = LabelEncoder()
all_labels = y_train_augmented + y_val
label_encoder.fit(all_labels)
y_train_encoded = label_encoder.transform(y_train_augmented)
y_val_encoded = label_encoder.transform(y_val)

num_classes = len(label_encoder.classes_)
print(f"类别数量: {num_classes}")
print(f"类别映射: {dict(zip(label_encoder.classes_, range(num_classes)))}")

# 计算类别权重（处理类别不平衡）
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)
class_weights = torch.FloatTensor(class_weights).to(device)
print("类别权重:", class_weights)

# 将标记转换为索引并填充/截断序列
def tokens_to_indices(tokens, word_to_idx, max_length):
    """将标记转换为索引并填充/截断到max_length"""
    indices = [word_to_idx.get(token, word_to_idx["<UNK>"]) for token in tokens[:max_length]]
    if len(indices) < max_length:
        indices += [word_to_idx["<PAD>"]] * (max_length - len(indices))
    return indices

X_train_indices = [tokens_to_indices(tokens, word_to_idx, MAX_SEQ_LENGTH) for tokens in X_train_augmented]
X_val_indices = [tokens_to_indices(tokens, word_to_idx, MAX_SEQ_LENGTH) for tokens in X_val]

# 定义自定义数据集
class MBTIDataset(Dataset):
    def __init__(self, indices, labels):
        self.indices = indices
        self.labels = labels
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return {
            'text': torch.tensor(self.indices[idx], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# 创建数据集和数据加载器
train_dataset = MBTIDataset(X_train_indices, y_train_encoded)
val_dataset = MBTIDataset(X_val_indices, y_val_encoded)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# 加载预训练的GloVe嵌入（如果可用）
def load_glove_embeddings(word_to_idx, embedding_dim, glove_path):
    """加载GloVe预训练词嵌入"""
    try:
        # 创建嵌入矩阵
        embedding_matrix = np.zeros((len(word_to_idx), embedding_dim))
        
        # 读取GloVe文件
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                if word in word_to_idx:
                    vector = np.array(values[1:], dtype='float32')
                    embedding_matrix[word_to_idx[word]] = vector
        
        return embedding_matrix
    except Exception as e:
        print(f"加载GloVe嵌入时出错: {e}")
        print("将使用随机初始化的嵌入")
        return None

# 注意力机制类
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, lstm_output):
        # lstm_output: [batch_size, seq_len, hidden_dim*2]
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)
        # 计算上下文向量
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights

# 定义改进的RNN模型（带注意力机制）
class ImprovedRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout, pad_idx, pretrained_embeddings=None):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # 如果有预训练嵌入，使用它
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            
        self.rnn = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        
        # 添加注意力层
        self.attention = AttentionLayer(hidden_dim)
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text = [batch size, seq len]
        embedded = self.dropout(self.embedding(text))
        # embedded = [batch size, seq len, embedding dim]
        
        # 通过LSTM
        output, (hidden, cell) = self.rnn(embedded)
        # output = [batch size, seq len, hidden dim * 2]
        
        # 应用注意力机制
        attention_output, attention_weights = self.attention(output)
        # attention_output = [batch size, hidden dim * 2]
        
        # 通过全连接层
        x = F.relu(self.fc1(attention_output))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x, attention_weights

# 尝试加载GloVe嵌入
embedding_matrix = None
if os.path.exists(GLOVE_PATH):
    print("加载GloVe预训练词嵌入...")
    embedding_matrix = load_glove_embeddings(word_to_idx, EMBEDDING_DIM, GLOVE_PATH)

# 初始化模型
model = ImprovedRNN(
    vocab_size=len(vocab),
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=num_classes,
    n_layers=NUM_LAYERS,
    dropout=DROPOUT,
    pad_idx=word_to_idx["<PAD>"],
    pretrained_embeddings=embedding_matrix
)

model = model.to(device)

# 定义带类别权重的损失函数和优化器
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# 训练函数
def train(model, dataloader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    all_predictions = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="训练中"):
        optimizer.zero_grad()
        
        text = batch['text'].to(device)
        labels = batch['label'].to(device)
        
        predictions, _ = model(text)
        
        loss = criterion(predictions, labels)
        
        # 计算准确率
        _, predicted = torch.max(predictions, 1)
        correct = (predicted == labels).float().sum()
        acc = correct / len(labels)
        
        # 收集预测和标签以便后续计算详细指标
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader), all_predictions, all_labels

# 评估函数
def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    all_predictions = []
    all_labels = []
    all_attention_weights = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估中"):
            text = batch['text'].to(device)
            labels = batch['label'].to(device)
            
            predictions, attention_weights = model(text)
            
            loss = criterion(predictions, labels)
            
            # 计算准确率
            _, predicted = torch.max(predictions, 1)
            correct = (predicted == labels).float().sum()
            acc = correct / len(labels)
            
            # 收集预测和标签以便后续计算详细指标
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 收集注意力权重
            all_attention_weights.append(attention_weights.cpu().numpy())
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader), all_predictions, all_labels, all_attention_weights

# 计算每个类别的精确率、召回率和F1分数
def calculate_metrics(y_true, y_pred, label_encoder):
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    
    # 计算每个类别的精确率、召回率和F1分数
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    # 获取类别名称
    class_names = label_encoder.classes_
    
    # 创建包含指标的字典
    metrics = {}
    for i, class_name in enumerate(class_names):
        metrics[class_name] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i]
        }
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    return metrics, cm

# 训练循环
print("开始训练...")
train_losses = []
train_accs = []
val_losses = []
val_accs = []
best_val_loss = float('inf')
early_stop_counter = 0

for epoch in range(NUM_EPOCHS):
    print(f"轮次 {epoch+1}/{NUM_EPOCHS}")
    
    # 训练模型
    train_loss, train_acc, train_preds, train_labels = train(model, train_loader, optimizer, criterion, CLIP)
    val_loss, val_acc, val_preds, val_labels, val_attention = evaluate(model, val_loader, criterion)
    
    # 学习率调度器更新
    scheduler.step(val_loss)
    
    # 记录损失和准确率
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    # 打印训练结果
    print(f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc*100:.2f}%")
    print(f"验证损失: {val_loss:.4f} | 验证准确率: {val_acc*100:.2f}%")
    
    # 计算并显示每个类别的指标
    if epoch % 5 == 0 or epoch == NUM_EPOCHS - 1:  # 每5轮或最后一轮显示详细指标
        print("\n计算详细评估指标...")
        val_metrics, val_cm = calculate_metrics(val_labels, val_preds, label_encoder)
        
        print("每个类别的指标:")
        for class_name, metrics in val_metrics.items():
            print(f"{class_name}: 精确率={metrics['precision']:.4f}, 召回率={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
    
    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'word_to_idx': word_to_idx,
            'label_encoder': label_encoder,
            'config': {
                'max_seq_length': MAX_SEQ_LENGTH,
                'embedding_dim': EMBEDDING_DIM,
                'hidden_dim': HIDDEN_DIM,
                'num_layers': NUM_LAYERS,
                'dropout': DROPOUT
            }
        }, os.path.join(MODEL_SAVE_PATH, 'best_model.pt'))
        print(f"保存最佳模型，验证损失: {val_loss:.4f}")
    else:
        early_stop_counter += 1
        if early_stop_counter >= EARLY_STOP_PATIENCE:
            print(f"早停：验证损失在{EARLY_STOP_PATIENCE}轮内未改善")
            break

# 保存最终模型
torch.save({
    'epoch': epoch + 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_loss': val_loss,
    'val_acc': val_acc,
    'word_to_idx': word_to_idx,
    'label_encoder': label_encoder,
    'config': {
        'max_seq_length': MAX_SEQ_LENGTH,
        'embedding_dim': EMBEDDING_DIM,
        'hidden_dim': HIDDEN_DIM,
        'num_layers': NUM_LAYERS,
        'dropout': DROPOUT
    }
}, os.path.join(MODEL_SAVE_PATH, 'final_model.pt'))
print(f"在{epoch+1}轮后保存最终模型")

# 绘制训练和验证指标
plt.figure(figsize=(16, 12))

plt.subplot(2, 2, 1)
plt.plot(train_losses, label='训练损失')
plt.plot(val_losses, label='验证损失')
plt.xlabel('轮次')
plt.ylabel('损失')
plt.legend()
plt.title('损失 vs. 轮次')

plt.subplot(2, 2, 2)
plt.plot(train_accs, label='训练准确率')
plt.plot(val_accs, label='验证准确率')
plt.xlabel('轮次')
plt.ylabel('准确率')
plt.legend()
plt.title('准确率 vs. 轮次')

# 如果有验证集混淆矩阵，则绘制
if 'val_cm' in locals():
    plt.subplot(2, 2, 3)
    plt.imshow(val_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('混淆矩阵')
    plt.colorbar()
    
    # 设置刻度标签
    classes = label_encoder.classes_
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    
    # 在每个单元格中显示值
    thresh = val_cm.max() / 2.
    for i in range(val_cm.shape[0]):
        for j in range(val_cm.shape[1]):
            plt.text(j, i, format(val_cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if val_cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')

# 可视化注意力权重（针对一个示例）
if len(val_attention) > 0:
    plt.subplot(2, 2, 4)
    # 获取第一个批次的第一个样本
    sample_attention = val_attention[0][0]
    plt.imshow(sample_attention.T, aspect='auto', cmap='hot')
    plt.title('注意力可视化（样本）')
    plt.xlabel('词索引')
    plt.ylabel('注意力权重')
    plt.colorbar()

plt.tight_layout()
plt.savefig(os.path.join(MODEL_SAVE_PATH, 'training_metrics.png'))
plt.show()

print("训练完成!")
print(f"模型和训练指标已保存到: {MODEL_SAVE_PATH}")

# 预测MBTI类型的函数
def predict_mbti(text, model, word_to_idx, label_encoder, max_length, device):
    """预测新文本的MBTI类型"""
    # 清洗和分词文本
    cleaned_text = clean_text(text)
    tokens = tokenize_text(cleaned_text)
    
    # 将标记转换为索引
    indices = tokens_to_indices(tokens, word_to_idx, max_length)
    
    # 转换为张量并添加批次维度
    tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    
    # 设置模型为评估模式
    model.eval()
    
    # 获取预测
    with torch.no_grad():
        prediction, attention_weights = model(tensor)
        probs = F.softmax(prediction, dim=1)
        conf, predicted_class = torch.max(probs, 1)
        predicted_type = label_encoder.inverse_transform(predicted_class.cpu().numpy())[0]
    
    return predicted_type, conf.item(), attention_weights.squeeze().cpu().numpy()

# 示例：使用模型进行预测
def prediction_example():
    # 加载最佳模型
    checkpoint = torch.load(os.path.join(MODEL_SAVE_PATH, 'best_model.pt'))
    
    # 重新创建模型结构
    model = ImprovedRNN(
        vocab_size=len(word_to_idx),
        embedding_dim=checkpoint['config']['embedding_dim'],
        hidden_dim=checkpoint['config']['hidden_dim'],
        output_dim=num_classes,
        n_layers=checkpoint['config']['num_layers'],
        dropout=checkpoint['config']['dropout'],
        pad_idx=word_to_idx["<PAD>"]
    )
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # 示例文本
    sample_text = "I love exploring new ideas and connecting with people. I enjoy thinking about the future and finding creative solutions to problems."
    
    # 预测MBTI类型
    predicted_type, confidence, attention = predict_mbti(
        sample_text, model, word_to_idx, label_encoder, MAX_SEQ_LENGTH, device
    )
    
    print(f"\n示例文本: '{sample_text}'")
    print(f"预测的MBTI类型: {predicted_type} (置信度: {confidence:.4f})")
    
    # 可视化注意力权重
    tokens = tokenize_text(clean_text(sample_text))[:MAX_SEQ_LENGTH]
    
    # 只显示有意义的词和对应的注意力权重
    meaningful_tokens = []
    meaningful_weights = []
    
    for i, token in enumerate(tokens):
        if i < len(attention):
            meaningful_tokens.append(token)
            meaningful_weights.append(attention[i])
    
    # 创建词-注意力对，并按注意力降序排序
    token_attention_pairs = sorted(
        zip(meaningful_tokens, meaningful_weights),
        key=lambda x: x[1],
        reverse=True
    )
    
    # 显示前10个最重要的词
    print("\n最有影响力的单词 (按注意力权重排序):")
    for token, weight in token_attention_pairs[:10]:
        print(f"{token}: {weight:.4f}")

# 如果这是主程序，运行示例预测
if __name__ == "__main__":
    # 训练结束后运行一个预测示例
    prediction_example()