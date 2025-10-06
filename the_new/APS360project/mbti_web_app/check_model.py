import pickle
import os

MODEL_DIR = r"C:\APS360_project\MBTI-Prediction-\the_new\APS360project\mbti_web_app\models\text\ml"

# 加载模型
with open(os.path.join(MODEL_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)

# 加载向量化器
with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

# 加载标签编码器
with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

print("✅ 模型、向量化器、标签编码器加载成功！")
print("模型类型:", type(model))
