import os
import shutil
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

# Define source and target paths
source_path = r"C:\Users\lnasl\Desktop\APS360\APS360\Data\Face_emotion\Combined"
target_path = r"C:\Users\lnasl\Desktop\APS360\APS360\Data\Preprocess"

# Create directories for train, validation, and test sets
os.makedirs(os.path.join(target_path, "train"), exist_ok=True)
os.makedirs(os.path.join(target_path, "val"), exist_ok=True)
os.makedirs(os.path.join(target_path, "test"), exist_ok=True)

# Get the list of emotion categories
emotion_categories = [d for d in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, d))]
print(f"Emotion categories: {emotion_categories}")

# Create emotion subdirectories inside each split directory
for split in ["train", "val", "test"]:
    for emotion in emotion_categories:
        os.makedirs(os.path.join(target_path, split, emotion), exist_ok=True)

# Count the number of images in each category and visualize
emotion_counts = {}
for emotion in emotion_categories:
    emotion_path = os.path.join(source_path, emotion)
    files = [f for f in os.listdir(emotion_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    emotion_counts[emotion] = len(files)

print("Number of images per emotion category:")
for emotion, count in emotion_counts.items():
    print(f"{emotion}: {count}")

# Visualize dataset imbalance
plt.figure(figsize=(10, 6))
plt.bar(emotion_counts.keys(), emotion_counts.values())
plt.title('Emotion Category Distribution')
plt.xlabel('Emotion Category')
plt.ylabel('Number of Images')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(target_path, 'emotion_distribution.png'))

# Split and copy images into train/val/test sets (8:1:1 ratio)
for emotion in emotion_categories:
    emotion_path = os.path.join(source_path, emotion)
    image_files = [f for f in os.listdir(emotion_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(image_files)  # Shuffle files randomly
    
    num_samples = len(image_files)
    num_train = int(num_samples * 0.8)
    num_val = int(num_samples * 0.1)
    
    # Split into training, validation, and test sets
    train_files = image_files[:num_train]
    val_files = image_files[num_train:num_train+num_val]
    test_files = image_files[num_train+num_val:]
    
    print(f"\n{emotion} split results:")
    print(f"Train: {len(train_files)}")
    print(f"Validation: {len(val_files)}")
    print(f"Test: {len(test_files)}")
    
    # Copy files to respective directories
    for file_list, split_name in zip([train_files, val_files, test_files], ["train", "val", "test"]):
        for file in file_list:
            src_file = os.path.join(emotion_path, file)
            dest_file = os.path.join(target_path, split_name, emotion, file)
            shutil.copy2(src_file, dest_file)
            
    # Verify copy results
    train_copied = len(os.listdir(os.path.join(target_path, "train", emotion)))
    val_copied = len(os.listdir(os.path.join(target_path, "val", emotion)))
    test_copied = len(os.listdir(os.path.join(target_path, "test", emotion)))
    
    print(f"Copied files - Train: {train_copied}, Val: {val_copied}, Test: {test_copied}")

print("\nData preprocessing and splitting completed!")
