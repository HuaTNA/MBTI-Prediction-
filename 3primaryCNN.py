import os
import json
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR

from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps, ImageEnhance
import timm

# 设置随机种子确保可重复性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# ============= 通道注意力机制 =============
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

# ============= 空间注意力机制 =============
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return x * self.sigmoid(out)

# ============= 优化的情绪识别模型 =============
class BalancedEmotionNet(nn.Module):
    def __init__(self, num_classes=8, model_name='efficientnet_b0', dropout_rate=0.3, 
                 enable_attention=True):
        super().__init__()
        self.enable_attention = enable_attention
        
        # 加载预训练模型作为骨干网络
        self.backbone = timm.create_model(
            model_name, 
            pretrained=True, 
            num_classes=0,
            global_pool=''
        )
        
        # 获取特征维度
        if 'efficientnet' in model_name:
            feature_dim = 1280
        elif 'resnet50' in model_name:
            feature_dim = 2048
        else:
            feature_dim = 1280
        
        # 只冻结早期层，让模型有更多的学习能力
        if 'efficientnet' in model_name:
            # 只冻结前2个阶段的参数
            for name, param in self.backbone.named_parameters():
                if 'blocks.0.' in name or 'blocks.1.' in name:
                    param.requires_grad = False
        elif 'resnet' in model_name:
            # 只冻结前两层
            for name, param in self.backbone.named_parameters():
                if 'conv1.' in name or 'bn1.' in name:
                    param.requires_grad = False
        
        # 添加注意力机制
        if self.enable_attention:
            self.channel_attention = ChannelAttention(feature_dim, reduction=16)  # 降低reduction以增加表达能力
            self.spatial_attention = SpatialAttention(kernel_size=7)
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 平衡的分类器设计 - 中等深度和正则化
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        
        if self.enable_attention:
            features = self.channel_attention(features)
            features = self.spatial_attention(features)
        
        x = self.global_pool(features)
        x = self.classifier(x)
        
        return x

# ============= 数据加载和预处理 =============
def create_dataloaders(train_path, val_path, batch_size=64):
    """创建数据加载器，包括类别平衡采样器"""
    
    # 数据增强 - 中等强度
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    print("加载数据集...")
    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=val_transform)
    
    # 分析类别分布
    class_counts = {}
    for _, label in train_dataset.samples:
        class_name = train_dataset.classes[label]
        if class_name not in class_counts:
            class_counts[class_name] = 0
        class_counts[class_name] += 1
    
    # 计算类别权重（用于加权采样）
    class_weights = []
    samples_weight = []
    for idx, (_, label) in enumerate(train_dataset.samples):
        class_name = train_dataset.classes[label]
        count = class_counts[class_name]
        # 类别平衡权重：样本越少的类别权重越高
        weight = 1.0 / count
        class_weights.append(weight)
        samples_weight.append(weight)
    
    # 创建加权采样器
    sampler = WeightedRandomSampler(
        weights=samples_weight,
        num_samples=len(train_dataset),
        replacement=True
    )
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler,  # 使用加权采样器
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"训练集: {len(train_dataset)}张图片，{len(train_dataset.classes)}个类别")
    print(f"验证集: {len(val_dataset)}张图片")
    print("训练集类别分布:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}张图像")
    
    # 返回数据加载器和类别信息
    return train_loader, val_loader, train_dataset.classes, class_counts

# ============= 训练函数 =============
def train_model(train_path, val_path, model_name='efficientnet_b0', 
                num_epochs=100, batch_size=64, learning_rate=0.0005, 
                weight_decay=0.0001, dropout_rate=0.3, 
                results_dir=None):
    """训练平衡的情绪识别模型"""
    
    # 创建结果目录
    if results_dir is None:
        results_dir = f"balanced_emotion_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建数据加载器
    train_loader, val_loader, class_names, class_counts = create_dataloaders(
        train_path, val_path, batch_size
    )
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型
    model = BalancedEmotionNet(
        num_classes=len(class_names),
        model_name=model_name,
        dropout_rate=dropout_rate,
        enable_attention=True
    )
    model = model.to(device)
    
    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    # 计算类别权重（用于损失函数）
    class_weights = []
    num_samples = sum(class_counts.values())
    num_classes = len(class_names)
    
    for class_name in class_names:
        count = class_counts.get(class_name, 0)
        # 权重公式：总样本数/(类别数*该类样本数)
        weight = num_samples / (num_classes * count) if count > 0 else 1.0
        class_weights.append(weight)
    
    # 转换为张量
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    print("类别权重:")
    for i, (name, weight) in enumerate(zip(class_names, class_weights)):
        print(f"  {name}: {weight:.4f}")
    
    # 损失函数 - 带权重的交叉熵
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
    
    # 区分不同部分使用不同的学习率
    backbone_params = []
    attention_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name:
            backbone_params.append(param)
        elif 'attention' in name:
            attention_params.append(param)
        else:
            classifier_params.append(param)
    
    # 优化器 - 使用不同的学习率
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': learning_rate * 0.1},
        {'params': attention_params, 'lr': learning_rate * 0.5},
        {'params': classifier_params, 'lr': learning_rate}
    ], weight_decay=weight_decay)
    
    # 学习率调度器 - OneCycleLR
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[learning_rate * 0.1, learning_rate * 0.5, learning_rate],
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1000
    )
    
    # 混合精度训练
    scaler = GradScaler()
    
    # 记录训练历史
    history = []
    
    # 用于早停的变量
    best_val_acc = 0.0
    best_epoch = 0
    best_model_state = None
    patience = 15
    patience_counter = 0
    
    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # 进度条
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        # 类别级别的准确率统计
        train_class_correct = [0] * len(class_names)
        train_class_total = [0] * len(class_names)
        
        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 混合精度训练
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # 反向传播
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # 更新学习率
            scheduler.step()
            
            # 计算训练指标
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
            train_loss += loss.item()
            
            # 更新类别级别的准确率
            for i in range(targets.size(0)):
                label = targets[i]
                train_class_correct[label] += (predicted[i] == label).item()
                train_class_total[label] += 1
            
            # 更新进度条
            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100 * train_correct / train_total:.2f}%",
                'lr': f"{optimizer.param_groups[2]['lr']:.6f}"
            })
        
        # 计算平均训练指标
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = 100 * train_correct / train_total
        
        # 计算各个类别的训练准确率
        train_class_accuracies = {}
        for i in range(len(class_names)):
            if train_class_total[i] > 0:
                accuracy = 100 * train_class_correct[i] / train_class_total[i]
                train_class_accuracies[class_names[i]] = accuracy
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # 类别级别的准确率统计
        val_class_correct = [0] * len(class_names)
        val_class_total = [0] * len(class_names)
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            
            for inputs, targets in val_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
                val_loss += loss.item()
                
                # 更新类别级别的准确率
                for i in range(targets.size(0)):
                    label = targets[i]
                    val_class_correct[label] += (predicted[i] == label).item()
                    val_class_total[label] += 1
                
                val_pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100 * val_correct / val_total:.2f}%"
                })
        
        # 计算平均验证指标
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = 100 * val_correct / val_total
        
        # 计算各个类别的验证准确率
        val_class_accuracies = {}
        for i in range(len(class_names)):
            if val_class_total[i] > 0:
                accuracy = 100 * val_class_correct[i] / val_class_total[i]
                val_class_accuracies[class_names[i]] = accuracy
        
        # 记录历史
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_acc': avg_train_acc,
            'val_loss': avg_val_loss,
            'val_acc': avg_val_acc,
            'lr': optimizer.param_groups[2]['lr'],
            'train_class_accuracies': train_class_accuracies,
            'val_class_accuracies': val_class_accuracies
        })
        
        # 输出当前结果
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.2f}%")
        
        # 计算准确率差距
        gap = avg_train_acc - avg_val_acc
        print(f"准确率差距: {gap:.2f}% {'(可能过拟合)' if gap > 15 else ''}")
        
        # 输出各类别的验证准确率
        print("各类别验证准确率:")
        for class_name, accuracy in val_class_accuracies.items():
            print(f"  {class_name}: {accuracy:.2f}%")
        
        # 保存最佳模型
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_epoch = epoch + 1
            best_model_state = model.state_dict()
            patience_counter = 0
            
            # 保存模型
            model_path = os.path.join(results_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"保存最佳模型，验证准确率: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n早停: {patience}个epoch没有改进, 最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})")
                break
    
    # 训练结束，恢复最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"\n训练完成。加载最佳模型 (Epoch {best_epoch}), 验证准确率: {best_val_acc:.2f}%")
    
    # 保存训练历史
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(results_dir, "training_history.csv"), index=False)
    
    # 绘制训练历史图表
    plt.figure(figsize=(15, 10))
    
    # 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(history_df['epoch'], history_df['train_loss'], label='训练损失')
    plt.plot(history_df['epoch'], history_df['val_loss'], label='验证损失')
    plt.title('损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot(history_df['epoch'], history_df['train_acc'], label='训练准确率')
    plt.plot(history_df['epoch'], history_df['val_acc'], label='验证准确率')
    plt.title('准确率曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # 学习率曲线
    plt.subplot(2, 2, 3)
    plt.plot(history_df['epoch'], history_df['lr'])
    plt.title('学习率变化')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    
    # 类别准确率
    plt.subplot(2, 2, 4)
    
    # 获取最后一个epoch的各类别准确率
    last_val_accuracies = history[-1]['val_class_accuracies']
    classes = list(last_val_accuracies.keys())
    accuracies = [last_val_accuracies[cls] for cls in classes]
    
    # 按准确率排序
    sorted_indices = np.argsort(accuracies)
    sorted_classes = [classes[i] for i in sorted_indices]
    sorted_accuracies = [accuracies[i] for i in sorted_indices]
    
    plt.barh(sorted_classes, sorted_accuracies)
    plt.title('最终各类别验证准确率')
    plt.xlabel('Accuracy (%)')
    plt.tight_layout()
    
    plt.savefig(os.path.join(results_dir, "training_history.png"))
    
    # 保存最终模型和配置
    final_model_path = os.path.join(results_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    
    # 保存训练配置和结果
    config = {
        'model_name': model_name,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'dropout_rate': dropout_rate,
        'num_classes': len(class_names),
        'class_names': class_names,
        'best_val_acc': float(best_val_acc),
        'best_epoch': best_epoch,
        'final_val_acc': float(history[-1]['val_acc']),
        'final_train_acc': float(history[-1]['train_acc']),
        'class_accuracies': {k: float(v) for k, v in last_val_accuracies.items()},
        'train_path': train_path,
        'val_path': val_path,
        'best_model_path': os.path.join(results_dir, 'best_model.pth')
    }
    
    with open(os.path.join(results_dir, "training_config.json"), 'w') as f:
        json.dump(config, f, indent=4)
    
    return model, history_df, config

# ============= 预测函数 =============
def predict(model, image_path, class_names, device=None):
    """使用训练好的模型预测图像的情绪"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 转换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 加载和预处理图像
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # 预测
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(probs).item()
    
    # 返回结果
    result = {
        'emotion': class_names[predicted_class],
        'confidence': probs[predicted_class].item(),
        'probabilities': {
            class_names[i]: probs[i].item() 
            for i in range(len(class_names))
        }
    }
    
    return result

# ============= 主函数 =============
if __name__ == "__main__":
    # 训练配置
    config = {
        'train_path': "H:/CODE/APS360/Data/Face_emotion/Split/train",
        'val_path': "H:/CODE/APS360/Data/Face_emotion/Split/val",
        'model_name': 'efficientnet_b0',
        'num_epochs': 100,
        'batch_size': 16,
        'learning_rate': 0.0001,  # 较高的学习率，提高模型学习能力
        'weight_decay': 0.0001,   # 较低的权重衰减，防止欠拟合
        'dropout_rate': 0.4,      # 较低的dropout率，提高表达能力
        'results_dir': f"H:/CODE/APS360Model/balanced_emotion_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }
    
    # 训练模型
    model, history, training_config = train_model(**config)
    
    print("\n训练完成！")
    print(f"最佳验证准确率: {training_config['best_val_acc']:.2f}% (第{training_config['best_epoch']}轮)")
    print(f"模型已保存至: {training_config['best_model_path']}")
    
    # 打印各类别的准确率
    print("\n各类别准确率:")
    for class_name, accuracy in training_config['class_accuracies'].items():
        print(f"  {class_name}: {accuracy:.2f}%")