import cv2
import dlib
import numpy as np
import os
import random
from skimage.feature import hog, local_binary_pattern
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# **初始化 Dlib 模型**
predictor = dlib.shape_predictor("H:/CODE/APS360/Model/shape_predictor_68_face_landmarks.dat")

# **减少训练数据**
MAX_SAMPLES_PER_CATEGORY = 200  # 先用 200 张/类 训练 SVM

def extract_features(img_path):
    """ 读取图片 & 提取 HOG + LBP + 关键点特征 """
    image = cv2.imread(img_path)
    if image is None:
        print(f"无法加载 {img_path}，跳过！")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    detector = dlib.get_frontal_face_detector()
    rects = detector(gray, 1)

    if len(rects) == 0:
        print(f"未检测到人脸: {img_path}")
        return None

    shape = predictor(gray, rects[0])
    landmarks = np.array([(p.x, p.y) for p in shape.parts()]).flatten()

    # **优化 HOG 特征**
    hog_features = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                       cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)

    # **增加 LBP 特征**
    lbp_features = local_binary_pattern(gray, P=8, R=1).flatten()

    return np.hstack([hog_features, lbp_features, landmarks]) 

# **加载数据集**
data_dir = r"H:\CODE\APS360\Data\Face_emotion\Split\train"
categories = os.listdir(data_dir)

X = []
y = []
image_paths = []
labels = []

print(f"找到 {len(categories)} 个类别: {categories}")
for category in tqdm(categories, desc="Processing Categories"):  
    label = categories.index(category)
    category_path = os.path.join(data_dir, category)

    img_names = os.listdir(category_path)[:MAX_SAMPLES_PER_CATEGORY]  
    for img_name in img_names:
        img_path = os.path.join(category_path, img_name)
        image_paths.append(img_path)
        labels.append(label)  

print(f"使用小数据集，总共 {len(image_paths)} 张图片")

# **多线程处理**
print("开始并行提取特征...")
with ThreadPoolExecutor(max_workers=2) as executor:  
    futures = {executor.submit(extract_features, img_path): i for i, img_path in enumerate(image_paths)}

    for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting Features"):
        result = future.result()
        if result is not None:
            X.append(result)
            y.append(labels[futures[future]])

print("特征提取完成！")
print(f"样本数: {len(X)}, 标签数: {len(y)}")

# **数据集划分**
print("开始划分训练集和测试集...")
X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("数据集划分完成！")

# **调整 SVM 超参数**
print("开始训练 SVM...")
svm_model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=10, gamma='scale'))

for i in tqdm(range(1), desc="Training SVM"):
    svm_model.fit(X_train, y_train)

print("SVM 训练完成！")

# **评估模型**
print("计算训练准确率和测试准确率...")
train_acc = svm_model.score(X_train, y_train)
test_acc = svm_model.score(X_test, y_test)

print(f"Training Accuracy: {train_acc:.2%}")
print(f"Testing Accuracy: {test_acc:.2%}")
