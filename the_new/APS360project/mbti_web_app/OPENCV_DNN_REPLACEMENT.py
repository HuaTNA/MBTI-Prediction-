"""
这个文件包含用 OpenCV DNN 替换 MediaPipe 的代码

将这些函数替换到 app.py 中对应的位置
"""

# ===================================================================
# 1. 在文件顶部的全局变量部分，替换 face_detector 的初始化
# ===================================================================
# 找到这行：
# face_detector = None

# 替换为：
face_detector = None  # OpenCV DNN face detector
face_detector_confidence = 0.5  # 检测置信度阈值


# ===================================================================
# 2. 替换 initialize_face_detector() 函数
# ===================================================================
def initialize_face_detector():
    """初始化 OpenCV DNN 人脸检测器（替代 MediaPipe）"""
    global face_detector

    # 模型文件路径
    model_dir = os.path.join(os.path.dirname(__file__), 'models', 'face_detection')
    prototxt_path = os.path.join(model_dir, 'deploy.prototxt')
    model_path = os.path.join(model_dir, 'res10_300x300_ssd_iter_140000.caffemodel')

    # 检查模型文件是否存在
    if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"人脸检测模型文件不存在。请运行: python download_face_models.py\n"
            f"缺少文件:\n"
            f"  - {prototxt_path}\n"
            f"  - {model_path}"
        )

    # 加载 OpenCV DNN 模型
    face_detector = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    logger.info("✅ 成功初始化 OpenCV DNN 人脸检测器（替代 MediaPipe）")


# ===================================================================
# 3. 替换 detect_and_crop_face() 函数
# ===================================================================
def detect_and_crop_face(image, padding=0.2):
    """
    使用 OpenCV DNN 检测并裁剪图像中的人脸

    参数:
        image: 输入图像 (BGR)
        padding: 边界框周围的额外填充，表示为边界框大小的比例

    返回:
        成功时返回裁剪后的人脸图像和True，失败时返回原始图像和False
    """
    global face_detector, face_detector_confidence

    # 确保人脸检测器已初始化
    if face_detector is None:
        initialize_face_detector()

    h, w = image.shape[:2]

    # 预处理图像用于 DNN 模型
    # 创建 blob：resize to 300x300, subtract mean values
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    # 设置输入并进行前向传播
    face_detector.setInput(blob)
    detections = face_detector.forward()

    # 找到置信度最高的人脸
    best_confidence = 0
    best_box = None

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > face_detector_confidence and confidence > best_confidence:
            best_confidence = confidence
            # 获取边界框坐标
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            best_box = box.astype("int")

    # 如果没有检测到人脸
    if best_box is None:
        return image, False

    # 解包坐标
    x1, y1, x2, y2 = best_box

    # 添加填充
    box_width = x2 - x1
    box_height = y2 - y1
    x1 = max(0, int(x1 - box_width * padding))
    y1 = max(0, int(y1 - box_height * padding))
    x2 = min(w, int(x2 + box_width * padding))
    y2 = min(h, int(y2 + box_height * padding))

    # 裁剪人脸区域
    face_crop = image[y1:y2, x1:x2]

    # 确保裁剪的人脸不为空
    if face_crop.size == 0:
        return image, False

    return face_crop, True


# ===================================================================
# 使用说明
# ===================================================================
"""
替换步骤:

1. 在 app.py 中，找到 initialize_face_detector() 函数（大约在第81-88行）
2. 用上面的新 initialize_face_detector() 函数替换它

3. 找到 detect_and_crop_face() 函数（大约在第90-150行）
4. 用上面的新 detect_and_crop_face() 函数替换它

5. 在文件顶部，删除或注释掉 MediaPipe 相关的导入：
   # import mediapipe as mp
   # mp_face_detection = mp.solutions.face_detection

6. 添加全局变量（如果还没有）：
   face_detector_confidence = 0.5

完成后保存文件，运行 python app.py
"""
