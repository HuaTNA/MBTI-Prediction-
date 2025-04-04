from adaptive_emotion_recognition import EmotionRecognitionModel

# 加载已训练的模型
model = EmotionRecognitionModel(model_path="best_facial_emotion_model.pth")

# 预测单张图片
result = model.predict("face.jpg")
print(f"检测到的情绪: {result['emotion']}, 置信度: {result['confidence']:.2f}")

# 查看所有情绪概率
for emotion, prob in result['probabilities'].items():
    print(f"{emotion}: {prob:.4f}")