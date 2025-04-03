import os
import shutil
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

# 定义源路径和目标路径
source_path = r"C:\Users\lnasl\Desktop\APS360\APS360\Data\Face_emotion\Combined"
target_path = r"C:\Users\lnasl\Desktop\APS360\APS360\Data\Preprocess"

# 创建训练、验证和测试目录
os.makedirs(os.path.join(target_path, "train"), exist_ok=True)
os.makedirs(os.path.join(target_path, "val"), exist_ok=True)
os.makedirs(os.path.join(target_path, "test"), exist_ok=True)

# 获取情绪类别列表
emotion_categories = [d for d in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, d))]
print(f"情绪类别: {emotion_categories}")

# 在每个分割目录中创建情绪子目录
for split in ["train", "val", "test"]:
    for emotion in emotion_categories:
        os.makedirs(os.path.join(target_path, split, emotion), exist_ok=True)

# 统计每个类别的图像数量并可视化
emotion_counts = {}
for emotion in emotion_categories:
    emotion_path = os.path.join(source_path, emotion)
    files = [f for f in os.listdir(emotion_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    emotion_counts[emotion] = len(files)

print("各情绪类别的图像数量:")
for emotion, count in emotion_counts.items():
    print(f"{emotion}: {count}")

# 可视化不平衡数据分布
plt.figure(figsize=(10, 6))
plt.bar(emotion_counts.keys(), emotion_counts.values())
plt.title('情绪类别分布')
plt.xlabel('情绪类别')
plt.ylabel('图像数量')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(target_path, 'emotion_distribution.png'))

# 按照8:1:1的比例分割并复制图像到对应目录
for emotion in emotion_categories:
    emotion_path = os.path.join(source_path, emotion)
    image_files = [f for f in os.listdir(emotion_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(image_files)  # 随机打乱文件顺序
    
    num_samples = len(image_files)
    num_train = int(num_samples * 0.8)
    num_val = int(num_samples * 0.1)
    
    # 分割为训练、验证和测试集
    train_files = image_files[:num_train]
    val_files = image_files[num_train:num_train+num_val]
    test_files = image_files[num_train+num_val:]
    
    print(f"\n{emotion} 分割结果:")
    print(f"训练集: {len(train_files)}")
    print(f"验证集: {len(val_files)}")
    print(f"测试集: {len(test_files)}")
    
    # 复制文件到相应目录
    for file_list, split_name in zip([train_files, val_files, test_files], ["train", "val", "test"]):
        for file in file_list:
            src_file = os.path.join(emotion_path, file)
            dest_file = os.path.join(target_path, split_name, emotion, file)
            shutil.copy2(src_file, dest_file)
            
    # 验证复制是否成功
    train_copied = len(os.listdir(os.path.join(target_path, "train", emotion)))
    val_copied = len(os.listdir(os.path.join(target_path, "val", emotion)))
    test_copied = len(os.listdir(os.path.join(target_path, "test", emotion)))
    
    print(f"复制结果 - 训练集: {train_copied}, 验证集: {val_copied}, 测试集: {test_copied}")

print("\n数据预处理和分割完成!")