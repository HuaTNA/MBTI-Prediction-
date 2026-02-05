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

# Initialize Dlib face landmark predictor
predictor = dlib.shape_predictor("H:/CODE/APS360/Model/shape_predictor_68_face_landmarks.dat")

# Limit number of samples per category (to speed up training)
MAX_SAMPLES_PER_CATEGORY = 200  # Use 200 images per class for initial SVM training

def extract_features(img_path):
    """Read an image and extract HOG + LBP + facial landmark features."""
    image = cv2.imread(img_path)
    if image is None:
        print(f"Warning: Unable to load {img_path}, skipping.")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    detector = dlib.get_frontal_face_detector()
    rects = detector(gray, 1)

    if len(rects) == 0:
        print(f"Warning: No face detected in {img_path}")
        return None

    shape = predictor(gray, rects[0])
    landmarks = np.array([(p.x, p.y) for p in shape.parts()]).flatten()

    # Extract HOG features
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False
    )

    # Extract LBP features
    lbp_features = local_binary_pattern(gray, P=8, R=1).flatten()

    # Concatenate all features
    return np.hstack([hog_features, lbp_features, landmarks]) 

# Load dataset
data_dir = r"H:\CODE\APS360\Data\Face_emotion\Split\train"
categories = os.listdir(data_dir)

X = []
y = []
image_paths = []
labels = []

print(f"Found {len(categories)} categories: {categories}")
for category in tqdm(categories, desc="Processing Categories"):  
    label = categories.index(category)
    category_path = os.path.join(data_dir, category)

    img_names = os.listdir(category_path)[:MAX_SAMPLES_PER_CATEGORY]  
    for img_name in img_names:
        img_path = os.path.join(category_path, img_name)
        image_paths.append(img_path)
        labels.append(label)  

print(f"Using a reduced dataset with a total of {len(image_paths)} images")

# Parallel feature extraction
print("Starting parallel feature extraction...")
with ThreadPoolExecutor(max_workers=2) as executor:  
    futures = {executor.submit(extract_features, img_path): i for i, img_path in enumerate(image_paths)}

    for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting Features"):
        result = future.result()
        if result is not None:
            X.append(result)
            y.append(labels[futures[future]])

print("Feature extraction completed.")
print(f"Number of samples: {len(X)}, Number of labels: {len(y)}")

# Split dataset
print("Splitting data into training and testing sets...")
X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split completed.")

# Train SVM model
print("Training SVM classifier...")
svm_model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=10, gamma='scale'))

for _ in tqdm(range(1), desc="Training SVM"):
    svm_model.fit(X_train, y_train)

print("SVM training completed.")

# Evaluate model
print("Evaluating model performance...")
train_acc = svm_model.score(X_train, y_train)
test_acc = svm_model.score(X_test, y_test)

print(f"Training Accuracy: {train_acc:.2%}")
print(f"Testing Accuracy: {test_acc:.2%}")
