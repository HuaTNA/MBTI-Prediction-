import os
import shutil
import random
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # Windows
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置随机种子确保可复现性
random.seed(42)
np.random.seed(42)

# 设置数据集路径
data_dir = Path(r"H:/CODE/APS360/Data/Face_emotion/Combined")

# 目标存储路径
base_output_dir = Path(r"H:/CODE/APS360/Data/Face_emotion/Split")
train_dir = base_output_dir / "train"
val_dir = base_output_dir / "val"
test_dir = base_output_dir / "test"

# 创建目标文件夹
for folder in [train_dir, val_dir, test_dir]:
    folder.mkdir(parents=True, exist_ok=True)    

# 设定目标图像尺寸
target_size = (224, 224)

# 设置划分比例
split_ratio = [0.8, 0.1, 0.1]  # 训练集、验证集、测试集比例

# 最小样本数设置
min_val_samples = 100  # 每类验证集最小样本数
min_test_samples = 100  # 每类测试集最小样本数

# 最大样本数设置（用于平衡训练集）
max_train_samples_per_class = 4000  # 每类训练集最大样本数

# =============== 分析类别分布 ===============
print("分析数据集类别分布...")
class_counts = {}
class_images = {}

for category in data_dir.iterdir():
    if category.is_dir():
        images = list(category.glob("*.*"))
        class_counts[category.name] = len(images)
        class_images[category.name] = images
        print(f"类别 {category.name}: {len(images)} 张图像")

# 找出样本最少的类别
min_class = min(class_counts.items(), key=lambda x: x[1])
min_class_count = min_class[1]
print(f"样本最少的类别: {min_class[0]}, {min_class_count} 张图像")

# 计算平衡后每个类别应该分配的样本数量
val_samples_per_class = max(min_val_samples, int(min_class_count * split_ratio[1]))
test_samples_per_class = max(min_test_samples, int(min_class_count * split_ratio[2]))

# 计算训练集上限，确保小类别有足够的样本
max_train_per_class = max_train_samples_per_class

print("\n计划的数据集划分:")
print(f"验证集: 每类 {val_samples_per_class} 张图像")
print(f"测试集: 每类 {test_samples_per_class} 张图像")
print(f"训练集: 每类最多 {max_train_per_class} 张图像")

# =============== 可视化原始数据分布 ===============
categories = list(class_counts.keys())
counts = [class_counts[cat] for cat in categories]

plt.figure(figsize=(12, 6))
bars = plt.bar(categories, counts)
plt.title("原始数据集类别分布")
plt.ylabel("图像数量")
plt.xticks(rotation=45, ha="right")

# 添加数字标签
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height}',
             ha='center', va='bottom')

plt.tight_layout()
plt.savefig(base_output_dir / "original_distribution.png")

# =============== 执行平衡划分 ===============
print("\n开始执行平衡划分...")
# 记录每个类别实际分配的样本数
actual_allocation = {
    'train': {},
    'val': {},
    'test': {}
}

# 对每个类别进行划分
for category, images in class_images.items():
    random.shuffle(images)  # 打乱数据顺序
    
    # 确定当前类别的样本分配
    num_val = val_samples_per_class
    num_test = test_samples_per_class
    
    # 计算训练集样本数，考虑上限
    remaining = len(images) - num_val - num_test
    num_train = min(remaining, max_train_per_class)
    
    # 如果样本不足以满足验证和测试需求，调整分配
    if remaining < 0:
        print(f"警告: 类别 {category} 样本不足 {len(images)}张，少于验证+测试所需 {num_val + num_test}张")
        # 按比例缩小验证和测试集的分配
        total = num_val + num_test
        num_val = int(len(images) * (num_val / total))
        num_test = int(len(images) * (num_test / total))
        num_train = 0
    
    # 分割图像
    val_images = images[:num_val]
    test_images = images[num_val:num_val+num_test]
    train_images = images[num_val+num_test:num_val+num_test+num_train]
    
    # 记录实际分配的样本数
    actual_allocation['train'][category] = len(train_images)
    actual_allocation['val'][category] = len(val_images)
    actual_allocation['test'][category] = len(test_images)
    
    print(f"类别 {category} 划分: 训练集 {len(train_images)}, 验证集 {len(val_images)}, 测试集 {len(test_images)}")
    
    # 创建目标目录
    train_category_dir = train_dir / category
    val_category_dir = val_dir / category
    test_category_dir = test_dir / category
    
    for dir_path in [train_category_dir, val_category_dir, test_category_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 处理并保存图像
    def process_and_save(image_list, target_dir):
        for img_path in image_list:
            try:
                with Image.open(img_path) as img:
                    # 确保图像是 RGB 模式
                    img = img.convert("RGB")
                    
                    # 先尝试保持纵横比的情况下进行裁剪到方形
                    width, height = img.size
                    if width != height:
                        size = min(width, height)
                        left = (width - size) // 2
                        top = (height - size) // 2
                        right = left + size
                        bottom = top + size
                        img = img.crop((left, top, right, bottom))
                    
                    # 调整大小
                    img = img.resize(target_size, Image.LANCZOS)
                    
                    # 保存调整后的图像
                    save_path = target_dir / img_path.name
                    img.save(save_path)
            except Exception as e:
                print(f"处理图像时出错 {img_path}: {e}")
    
    # 处理不同集合的图像
    process_and_save(train_images, train_category_dir)
    process_and_save(val_images, val_category_dir)
    process_and_save(test_images, test_category_dir)

# =============== 可视化平衡后的分布 ===============
# 训练集分布
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
categories = list(actual_allocation['train'].keys())
counts = [actual_allocation['train'][cat] for cat in categories]
bars = plt.bar(categories, counts)
plt.title("平衡后训练集分布")
plt.ylabel("图像数量")
plt.xticks(rotation=45, ha="right")
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height}',
             ha='center', va='bottom')

# 验证集分布
plt.subplot(1, 3, 2)
counts = [actual_allocation['val'][cat] for cat in categories]
bars = plt.bar(categories, counts)
plt.title("平衡后验证集分布")
plt.ylabel("图像数量")
plt.xticks(rotation=45, ha="right")
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height}',
             ha='center', va='bottom')

# 测试集分布
plt.subplot(1, 3, 3)
counts = [actual_allocation['test'][cat] for cat in categories]
bars = plt.bar(categories, counts)
plt.title("平衡后测试集分布")
plt.ylabel("图像数量")
plt.xticks(rotation=45, ha="right")
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height}',
             ha='center', va='bottom')

plt.tight_layout()
plt.savefig(base_output_dir / "balanced_distribution.png")

# =============== 统计最终分布 ===============
# 计算每个数据集的总样本数
train_total = sum(actual_allocation['train'].values())
val_total = sum(actual_allocation['val'].values())
test_total = sum(actual_allocation['test'].values())
total = train_total + val_total + test_total

# 输出最终统计信息
print("\n最终数据集统计:")
print(f"总样本数: {total}张图像")
print(f"训练集: {train_total}张图像 ({train_total/total*100:.1f}%)")
print(f"验证集: {val_total}张图像 ({val_total/total*100:.1f}%)")
print(f"测试集: {test_total}张图像 ({test_total/total*100:.1f}%)")

# 检查类别平衡情况
train_min = min(actual_allocation['train'].values()) if actual_allocation['train'] else 0
train_max = max(actual_allocation['train'].values()) if actual_allocation['train'] else 0
imbalance_ratio = train_max / train_min if train_min > 0 else float('inf')

print(f"\n训练集类别平衡情况:")
print(f"最少样本类别: {train_min}张图像")
print(f"最多样本类别: {train_max}张图像")
print(f"不平衡比例: {imbalance_ratio:.2f}")

print("\n数据集均衡划分完成！")