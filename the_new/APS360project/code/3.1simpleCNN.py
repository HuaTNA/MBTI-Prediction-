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

# ============= 32层深度情绪识别网络 =============
class DeepEmotionNet32(nn.Module):
    def __init__(self, num_classes=7, filters_multiplier=1.0, dropout_rate=0.5):
        """
        32层深度面部表情识别CNN模型，采用残差连接结构
        
        参数:
            num_classes (int): 分类类别数
            filters_multiplier (float): 卷积层通道数的乘数，用于调整模型容量
            dropout_rate (float): Dropout层的丢弃率
        """
        super(DeepEmotionNet32, self).__init__()
        
        # 计算各层通道数
        f = filters_multiplier
        init_channels = max(16, int(32 * f))
        channels_1 = init_channels
        channels_2 = max(32, int(64 * f))
        channels_3 = max(64, int(128 * f))
        
        # Initial convolution block (2 layers)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=channels_1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels_1)
        
        # Residual block 1 (6 layers: 2 conv + 2 bn + 2 identity connections)
        self.conv2_1 = nn.Conv2d(in_channels=channels_1, out_channels=channels_1, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(channels_1)
        self.conv2_2 = nn.Conv2d(in_channels=channels_1, out_channels=channels_1, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(channels_1)
        
        # Downsample + Residual block 2 (7 layers: 3 conv + 3 bn + 1 identity)
        self.conv3_0 = nn.Conv2d(in_channels=channels_1, out_channels=channels_2, kernel_size=1, stride=2)
        self.bn3_0 = nn.BatchNorm2d(channels_2)
        self.conv3_1 = nn.Conv2d(in_channels=channels_1, out_channels=channels_2, kernel_size=3, padding=1, stride=2)
        self.bn3_1 = nn.BatchNorm2d(channels_2)
        self.conv3_2 = nn.Conv2d(in_channels=channels_2, out_channels=channels_2, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(channels_2)
        
        # Residual block 3 (6 layers: 2 conv + 2 bn + 2 identity)
        self.conv4_1 = nn.Conv2d(in_channels=channels_2, out_channels=channels_2, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(channels_2)
        self.conv4_2 = nn.Conv2d(in_channels=channels_2, out_channels=channels_2, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(channels_2)
        
        # Downsample + Residual block 4 (7 layers: 3 conv + 3 bn + 1 identity)
        self.conv5_0 = nn.Conv2d(in_channels=channels_2, out_channels=channels_3, kernel_size=1, stride=2)
        self.bn5_0 = nn.BatchNorm2d(channels_3)
        self.conv5_1 = nn.Conv2d(in_channels=channels_2, out_channels=channels_3, kernel_size=3, padding=1, stride=2)
        self.bn5_1 = nn.BatchNorm2d(channels_3)
        self.conv5_2 = nn.Conv2d(in_channels=channels_3, out_channels=channels_3, kernel_size=3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(channels_3)
        
        # Final layers (4 layers: 1 adaptive pool + 3 fully connected)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Linear(channels_3 * 7 * 7, 1024)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(512, num_classes)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """使用He初始化来初始化所有卷积和全连接层的权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual block 1
        identity = x
        out = F.relu(self.bn2_1(self.conv2_1(x)))
        out = self.bn2_2(self.conv2_2(out))
        out += identity
        out = F.relu(out)
        
        # Downsample + Residual block 2
        identity = self.bn3_0(self.conv3_0(out))
        out = F.relu(self.bn3_1(self.conv3_1(out)))
        out = self.bn3_2(self.conv3_2(out))
        out += identity
        out = F.relu(out)
        
        # Residual block 3
        identity = out
        out = F.relu(self.bn4_1(self.conv4_1(out)))
        out = self.bn4_2(self.conv4_2(out))
        out += identity
        out = F.relu(out)
        
        # Downsample + Residual block 4
        identity = self.bn5_0(self.conv5_0(out))
        out = F.relu(self.bn5_1(self.conv5_1(out)))
        out = self.bn5_2(self.conv5_2(out))
        out += identity
        out = F.relu(out)
        
        # Final layers
        out = self.adaptive_pool(out)
        out = torch.flatten(out, 1)
        out = F.relu(self.fc1(out))
        out = self.dropout1(out)
        out = F.relu(self.fc2(out))
        out = self.dropout2(out)
        out = self.fc3(out)
        
        return out
    
def train_deep32_model(train_path, val_path, 
                       num_epochs=50, 
                       batch_size=32, 
                       learning_rate=0.0001, 
                       weight_decay=0.0005, 
                       dropout_rate=0.5,
                       filters_multiplier=1.0,
                       results_dir=None):
    """32层深度情绪识别模型训练函数"""
    # 创建结果目录
    results_dir = f"C:/Users/lnasl/Desktop/APS360/APS360/Model/deep32_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    # 数据增强 - 稍微增强以适应深度模型
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),  # 增加旋转角度
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # 增加色彩变化
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
    
    # 创建32层深度模型
    model = DeepEmotionNet32(
        num_classes=num_classes, 
        filters_multiplier=filters_multiplier,
        dropout_rate=dropout_rate
    )
    model = model.to(device)
    
    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    # 损失函数 - 带标签平滑的交叉熵
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 优化器 - 使用AdamW
    # 区分不同层次使用不同的学习率
    early_layers_params = []
    middle_layers_params = []
    late_layers_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'conv1' in name or 'bn1' in name or 'conv2' in name or 'bn2' in name:
            early_layers_params.append(param)
        elif 'conv3' in name or 'bn3' in name or 'conv4' in name or 'bn4' in name:
            middle_layers_params.append(param)
        elif 'conv5' in name or 'bn5' in name:
            late_layers_params.append(param)
        else:
            classifier_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': early_layers_params, 'lr': learning_rate * 0.5},  # 浅层学习率较低
        {'params': middle_layers_params, 'lr': learning_rate * 0.75},  # 中间层学习率适中
        {'params': late_layers_params, 'lr': learning_rate},  # 深层学习率标准
        {'params': classifier_params, 'lr': learning_rate * 3}  # 分类器使用更高学习率
    ], weight_decay=weight_decay)
    
    # 学习率调度器 - 一循环学习率调度
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[learning_rate * 0.5, learning_rate * 0.75, learning_rate, learning_rate * 3], 
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
    
    # 计算总批次数用于进度百分比显示
    total_batches = len(train_loader) * num_epochs
    processed_batches = 0
    
    # 进度百分比格式化
    def progress_percentage(current, total):
        return f"[进度: {current}/{total} ({current/total*100:.2f}%)]"
    
    # 训练循环
    print("\n开始训练..." + progress_percentage(0, num_epochs))
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # 显示整体训练进度
        overall_progress = progress_percentage(epoch+1, num_epochs)
        epoch_header = f"Epoch {epoch+1}/{num_epochs} {overall_progress}"
        print(f"\n{'-'*len(epoch_header)}\n{epoch_header}\n{'-'*len(epoch_header)}")
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # 创建进度条
        train_pbar = tqdm(train_loader, desc=f"训练中 {progress_percentage(0, len(train_loader))}")
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # 更新学习率
            scheduler.step()
            
            # 计算指标
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
            train_loss += loss.item()
            
            # 更新进度计数
            processed_batches += 1
            
            # 更新训练进度条
            batch_progress = progress_percentage(batch_idx+1, len(train_loader))
            total_progress = progress_percentage(processed_batches, total_batches)
            
            train_pbar.set_description(f"训练中 {batch_progress}")
            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100 * train_correct / train_total:.2f}%",
                'lr': f"{optimizer.param_groups[-1]['lr']:.6f}",
                'total': total_progress
            })
        
        train_pbar.close()
        
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
            val_pbar = tqdm(val_loader, desc=f"验证中 {progress_percentage(0, len(val_loader))}")
            
            for batch_idx, (inputs, targets) in enumerate(val_loader):
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
                
                # 更新验证进度条
                batch_progress = progress_percentage(batch_idx+1, len(val_loader))
                val_pbar.set_description(f"验证中 {batch_progress}")
                val_pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100 * val_correct / val_total:.2f}%"
                })
            
            val_pbar.close()
        
        # 计算平均验证损失和准确率
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = 100 * val_correct / val_total
        
        # 计算每个类别的准确率
        class_accuracies = {}
        for i in range(num_classes):
            if class_total[i] > 0:
                accuracy = 100 * class_correct[i] / class_total[i]
                class_accuracies[train_dataset.classes[i]] = accuracy
        
        # 计算本轮用时
        epoch_time = time.time() - epoch_start_time
        estimated_remaining = epoch_time * (num_epochs - (epoch + 1))
        
        # 格式化时间
        def format_time(seconds):
            hours, remainder = divmod(seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        
        # 记录历史
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_acc': avg_train_acc,
            'val_loss': avg_val_loss,
            'val_acc': avg_val_acc,
            'lr': optimizer.param_groups[-1]['lr'],
            'class_accuracies': class_accuracies
        })
        
        # 输出当前结果，包含进度百分比和时间估计
        total_progress = progress_percentage(processed_batches, total_batches)
        print(f"\n总体进度: {total_progress}")
        print(f"本轮用时: {format_time(epoch_time)}, 预计剩余: {format_time(estimated_remaining)}")
        print(f"Epoch {epoch+1}/{num_epochs}: "
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
    
    # 计算总训练时间
    total_training_time = time.time() - start_time
    print(f"\n训练完成! 总用时: {format_time(total_training_time)}")
    
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
        'total_training_time': total_training_time,
        'model_params': {
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'dropout_rate': dropout_rate,
            'batch_size': batch_size,
            'num_classes': num_classes,
            'filters_multiplier': filters_multiplier
        },
        'best_model_path': os.path.join(results_dir, 'best_model.pth')
    }
    
    with open(os.path.join(results_dir, "training_summary.json"), 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\n训练完成！最佳验证准确率: {best_val_acc:.2f}% (第{best_epoch}轮)")
    print(f"最佳模型已保存至: {os.path.join(results_dir, 'best_model.pth')}")
    
    return model, summary, history_df


if __name__ == "__main__":
    import argparse
    import os
    import torch
    from datetime import datetime
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='面部表情识别深度学习模型')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict'],
                        help='运行模式: train (训练) 或 predict (预测)')
    parser.add_argument('--data_dir', type=str, default='C:/Users/lnasl/Desktop/APS360/APS360/Data/Preprocess',
                        help='数据集根目录，应包含train和val子目录')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='保存模型和结果的输出目录 (默认使用时间戳命名)')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='权重衰减')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout率')
    parser.add_argument('--filters_multiplier', type=float, default=1.0, help='卷积通道乘数')
    parser.add_argument('--model_path', type=str, default=None, 
                        help='预测模式下加载的模型路径，或训练模式下继续训练的模型路径')
    parser.add_argument('--image_path', type=str, default=None, 
                        help='预测模式下要分析的图像路径')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # 创建输出目录
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"C:/Users/lnasl/Desktop/APS360/APS360/Model/deep32_emotion_{timestamp}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 配置数据路径
    train_path = os.path.join(args.data_dir, 'train')
    val_path = os.path.join(args.data_dir, 'val')
    
    # 检查GPU可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 根据模式选择操作
    if args.mode == 'train':
        print(f"开始训练模式...")
        print(f"训练集路径: {train_path}")
        print(f"验证集路径: {val_path}")
        print(f"结果保存至: {args.output_dir}")
        print(f"训练参数:")
        print(f"  轮数: {args.epochs}")
        print(f"  批次大小: {args.batch_size}")
        print(f"  学习率: {args.lr}")
        print(f"  权重衰减: {args.weight_decay}")
        print(f"  Dropout率: {args.dropout}")
        print(f"  通道乘数: {args.filters_multiplier}")
        
        # 训练模型
        training_params = {
            'train_path': train_path,
            'val_path': val_path,
            'num_epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'dropout_rate': args.dropout,
            'filters_multiplier': args.filters_multiplier,
            'results_dir': args.output_dir
        }
        
        try:
            # 训练模型
            model, summary, history = train_deep32_model(**training_params)
            
            # 打印训练摘要
            print("\n模型训练摘要:")
            print(f"最佳验证准确率: {summary['best_val_acc']:.2f}% (第{summary['best_epoch']}轮)")
            print(f"最终训练-验证差距: {summary['train_val_gap']:.2f}%")
            
            # 保存完整的命令行参数以便复现
            import json
            with open(os.path.join(args.output_dir, "command_args.json"), 'w') as f:
                json.dump(vars(args), f, indent=4)
            
            print(f"\n训练完成！所有结果已保存至: {args.output_dir}")
            
        except Exception as e:
            print(f"训练过程中发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
    
    elif args.mode == 'predict':
        if args.model_path is None or args.image_path is None:
            parser.error("预测模式需要同时提供 --model_path 和 --image_path 参数")
        
        if not os.path.exists(args.model_path):
            parser.error(f"模型文件不存在: {args.model_path}")
        
        if not os.path.exists(args.image_path):
            parser.error(f"图像文件不存在: {args.image_path}")
        
        print(f"开始预测模式...")
        print(f"加载模型: {args.model_path}")
        print(f"分析图像: {args.image_path}")
        
        try:
            # 获取模型的类别信息（假设在训练集中可获取）
            from torchvision import datasets
            try:
                train_dataset = datasets.ImageFolder(root=train_path)
                classes = train_dataset.classes
                num_classes = len(classes)
                print(f"从训练集获取类别信息: {classes}")
            except Exception:
                print("无法从训练集获取类别信息，将使用默认7类")
                classes = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
                num_classes = 7
            
            # 创建模型实例
            model = DeepEmotionNet32(
                num_classes=num_classes,
                filters_multiplier=args.filters_multiplier,
                dropout_rate=0.0  # 预测时关闭dropout
            )
            
            # 加载预训练权重
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            model.to(device)
            
            # 进行预测
            prediction_idx, confidence = predict_emotion(model, args.image_path, device)
            
            # 输出预测结果
            prediction_label = classes[prediction_idx]
            print(f"\n预测结果:")
            print(f"表情类别: {prediction_label}")
            print(f"置信度: {confidence*100:.2f}%")
            
            # 保存预测结果
            result_path = os.path.join(args.output_dir, "prediction_result.txt")
            with open(result_path, 'w') as f:
                f.write(f"图像: {args.image_path}\n")
                f.write(f"模型: {args.model_path}\n")
                f.write(f"预测类别: {prediction_label}\n")
                f.write(f"置信度: {confidence*100:.2f}%\n")
            
            print(f"\n预测结果已保存至: {result_path}")
            
        except Exception as e:
            print(f"预测过程中发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
    
    else:
        parser.error(f"未知模式: {args.mode}")