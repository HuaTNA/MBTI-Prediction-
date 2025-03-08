import os
import shutil
import random
from pathlib import Path
from PIL import Image

# 设置数据集路径
data_dir = Path(r"H:\CODE\APS360\Data\Face_emotion\Combined")

# 目标存储路径
base_output_dir = Path(r"H:\CODE\APS360\Data\Face_emotion\Split")
train_dir = base_output_dir / "train"
val_dir = base_output_dir / "val"
test_dir = base_output_dir / "test"

# 创建目标文件夹
for folder in [train_dir, val_dir, test_dir]:
    folder.mkdir(parents=True, exist_ok=True)

# 设定目标图像尺寸（例如 224x224）
target_size = (224, 224)

# 设置划分比例
split_ratio = [0.8, 0.1, 0.1]  # 训练集、验证集、测试集比例

# 遍历类别文件夹并进行划分
for category in data_dir.iterdir():
    if category.is_dir():
        images = list(category.glob("*.*"))  # 获取所有图片
        random.shuffle(images)  # 打乱数据顺序

        # 计算各数据集的样本数
        num_train = int(len(images) * split_ratio[0])
        num_val = int(len(images) * split_ratio[1])
        
        train_images = images[:num_train]
        val_images = images[num_train:num_train + num_val]
        test_images = images[num_train + num_val:]

        # 创建类别子文件夹并复制并调整大小
        for split, split_images in zip([train_dir, val_dir, test_dir], [train_images, val_images, test_images]):
            split_category_dir = split / category.name
            split_category_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in split_images:
                try:
                    with Image.open(img_path) as img:
                        img = img.convert("RGB")  # 确保图像是 RGB 模式
                        img = img.resize(target_size, Image.LANCZOS)  # 调整大小
                        save_path = split_category_dir / img_path.name
                        img.save(save_path)  # 保存调整后的图像
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

print("数据集划分并调整大小完成！")
