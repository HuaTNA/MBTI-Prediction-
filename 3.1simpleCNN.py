import os
import json
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR

from torchvision import datasets, transforms, models
from PIL import Image

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

# ============= 超简化模型架构 =============
class UltraMinimalEmotionNet(nn.Module):
    def __init__(self, num_classes=7, dropout_rate=0.5):
        super().__init__()
        
        # 使用MobileNetV3 Small，这是一个非常轻量级的模型
        self.backbone = models.mobilenet_v3_small(pretrained=True)
        
        # 冻结大部分层，只训练最后几层
        # MobileNetV3有16个block，冻结前14个
        for name, param in self.backbone.named_parameters():
            if 'features.14' not in name and 'features.15' not in name and 'classifier' not in name:
                param.requires_grad = False
        
        # 替换分类器
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, num_classes)
        )
        
        # 添加dropout作为单独层，方便调整
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.backbone.classifier(x)
        return x

# ============= 训练函数 =============
def train_ultra_minimal_model(train_path, val_path, 
                              num_epochs=50, 
                              batch_size=32, 
                              learning_rate=0.00005, 
                              weight_decay=0.001, 
                              dropout_rate=0.6,
                              results_dir=None):
    """超简化情绪识别模型训练"""
    # 创建结果目录

    results_dir = f"H:/CODE/APS360/Model/ultra_minimal_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    os.makedirs(results_dir, exist_ok=True)
    
    # 数据增强 - 简化以降低复杂性
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        # 添加随机旋转但幅度较小
        transforms.RandomRotation(5),
        # 很小的色彩变化
        transforms.ColorJitter(brightness=0.05, contrast=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    print("加载数据集...")
    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=val_transform)
    
    # 获取类别数
    num_classes = len(train_dataset.classes)
    print(f"检测到{num_classes}个类别")
    print(f"训练集: {len(train_dataset)}张图片")
    print(f"验证集: {len(val_dataset)}张图片")
    
    # 分析类别分布
    class_counts = {}
    for _, label in train_dataset.samples:
        class_name = train_dataset.classes[label]
        if class_name not in class_counts:
            class_counts[class_name] = 0
        class_counts[class_name] += 1
    
    print("训练集类别分布:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}张图像")
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
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
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型
    model = UltraMinimalEmotionNet(num_classes=num_classes, dropout_rate=dropout_rate)
    model = model.to(device)
    
    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    # 损失函数 - 带标签平滑的交叉熵
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 优化器 - 使用AdamW，区分冻结和非冻结参数
    # 区分不同部分使用不同的学习率
    feature_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'classifier' in name:
                classifier_params.append(param)
            else:
                feature_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': feature_params, 'lr': learning_rate},
        {'params': classifier_params, 'lr': learning_rate * 5}  # 分类器使用5倍学习率
    ], weight_decay=weight_decay)
    
    # 学习率调度器 - 一循环学习率调度
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[learning_rate, learning_rate * 5],  # 针对两组参数的最大学习率
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,       # 30%的时间用于预热
        div_factor=25,       # 初始学习率 = max_lr / 25
        final_div_factor=10000,  # 最终学习率 = 初始学习率 / 10000
        anneal_strategy='cos'
    )
    
    # 梯度缩放器（用于混合精度训练）
    scaler = GradScaler()
    
    # 初始化训练指标
    best_val_acc = 0.0
    best_epoch = 0
    history = []
    
    # 早停参数
    patience = 10
    patience_counter = 0
    
    # 训练循环
    print("\n开始训练...")
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # 创建进度条
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 混合精度训练
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # 反向传播
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # 梯度裁剪 - 使用较小的阈值防止梯度爆炸
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            scaler.step(optimizer)
            scaler.update()
            
            # 更新学习率
            scheduler.step()
            
            # 计算指标
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
            train_loss += loss.item()
            
            # 更新进度条
            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100 * train_correct / train_total:.2f}%",
                'lr': f"{optimizer.param_groups[1]['lr']:.6f}"
            })
        
        # 计算平均训练损失和准确率
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = 100 * train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # 每个类别的准确率统计
        class_correct = [0] * num_classes
        class_total = [0] * num_classes
        
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
                
                # 计算每个类别的准确率
                for i in range(targets.size(0)):
                    label = targets[i]
                    class_correct[label] += (predicted[i] == label).item()
                    class_total[label] += 1
                
                val_pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100 * val_correct / val_total:.2f}%"
                })
        
        # 计算平均验证损失和准确率
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = 100 * val_correct / val_total
        
        # 计算每个类别的准确率
        class_accuracies = {}
        for i in range(num_classes):
            if class_total[i] > 0:
                accuracy = 100 * class_correct[i] / class_total[i]
                class_accuracies[train_dataset.classes[i]] = accuracy
        
        # 记录历史
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_acc': avg_train_acc,
            'val_loss': avg_val_loss,
            'val_acc': avg_val_acc,
            'lr': optimizer.param_groups[1]['lr'],
            'class_accuracies': class_accuracies
        })
        
        # 输出当前结果
        print(f"\nEpoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.2f}%, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.2f}%")
        
        # 检查过拟合
        gap = avg_train_acc - avg_val_acc
        print(f"准确率差距: {gap:.2f}% {'(可能过拟合)' if gap > 10 else ''}")
        
        # 打印每个类别的准确率
        print("各类别验证准确率:")
        for class_name, accuracy in class_accuracies.items():
            print(f"  {class_name}: {accuracy:.2f}%")
        
        # 保存最佳模型
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            
            # 保存模型
            model_path = os.path.join(results_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"保存最佳模型，验证准确率: {best_val_acc:.2f}%")
            
            # 如果gap很小，额外保存一个模型版本
            if gap < 5:
                balanced_model_path = os.path.join(results_dir, f'balanced_model_gap{gap:.1f}.pth')
                torch.save(model.state_dict(), balanced_model_path)
                print(f"保存平衡模型，训练-验证差距仅为: {gap:.2f}%")
        else:
            patience_counter += 1
        
        # 早停检查
        if patience_counter >= patience:
            print(f"\n早停: {patience}个epoch没有改进, 最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})")
            break
    
    # 绘制训练历史
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(results_dir, "training_history.csv"), index=False)
    
    plt.figure(figsize=(12, 10))
    
    # 损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(history_df['epoch'], history_df['train_loss'], label='训练损失')
    plt.plot(history_df['epoch'], history_df['val_loss'], label='验证损失')
    plt.title('损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 准确率曲线
    plt.subplot(2, 1, 2)
    plt.plot(history_df['epoch'], history_df['train_acc'], label='训练准确率')
    plt.plot(history_df['epoch'], history_df['val_acc'], label='验证准确率')
    plt.title('准确率曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "training_history.png"))
    
    # 绘制每个类别的准确率变化
    plt.figure(figsize=(15, 8))
    for class_name in train_dataset.classes:
        class_acc = [epoch_data.get('class_accuracies', {}).get(class_name, 0) for epoch_data in history]
        plt.plot(range(1, len(class_acc)+1), class_acc, label=class_name)
    
    plt.title('各类别验证准确率变化')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "class_accuracies.png"))
    
    # 保存最后一个模型
    final_model_path = os.path.join(results_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    
    # 保存训练结果摘要
    final_val_acc = history_df['val_acc'].iloc[-1]
    final_train_acc = history_df['train_acc'].iloc[-1]
    
    summary = {
        'best_val_acc': float(best_val_acc),
        'best_epoch': best_epoch,
        'final_val_acc': float(final_val_acc),
        'final_train_acc': float(final_train_acc),
        'train_val_gap': float(final_train_acc - final_val_acc),
        'model_params': {
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'dropout_rate': dropout_rate,
            'batch_size': batch_size,
            'num_classes': num_classes
        },
        'best_model_path': os.path.join(results_dir, 'best_model.pth')
    }
    
    with open(os.path.join(results_dir, "training_summary.json"), 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\n训练完成！最佳验证准确率: {best_val_acc:.2f}% (第{best_epoch}轮)")
    print(f"最佳模型已保存至: {os.path.join(results_dir, 'best_model.pth')}")
    
    return model, summary, history_df

# ============= 预测函数 =============
def predict_emotion(model, image_path, device=None):
    """使用训练好的模型预测情绪"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model.eval()
    model = model.to(device)
    
    # 转换与验证集相同
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
    
    return prediction.item(), confidence.item()

# ============= 主函数 =============
if __name__ == "__main__":
    # 训练参数
    params = {
        'train_path': "H:/CODE/APS360/Data/Face_emotion/Split/train",
        'val_path': "H:/CODE/APS360/Data/Face_emotion/Split/val",
        'num_epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.00005,
        'weight_decay': 0.001,
        'dropout_rate': 0.6,
        'results_dir': f"ultra_minimal_emotion_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }
    
    # 训练模型
    model, summary, history = train_ultra_minimal_model(**params)
    
    print("\n模型训练摘要:")
    print(f"最佳验证准确率: {summary['best_val_acc']:.2f}% (第{summary['best_epoch']}轮)")
    print(f"最终训练-验证差距: {summary['train_val_gap']:.2f}%")