"""
下载 OpenCV DNN 人脸检测模型文件
"""

import os
import urllib.request

# 创建模型目录
models_dir = os.path.join(os.path.dirname(__file__), 'models', 'face_detection')
os.makedirs(models_dir, exist_ok=True)

# 模型文件 URLs（OpenCV 官方提供的预训练模型）
models = {
    'deploy.prototxt': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt',
    'res10_300x300_ssd_iter_140000.caffemodel': 'https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel'
}

print("正在下载 OpenCV DNN 人脸检测模型...")
print("="*60)

for filename, url in models.items():
    filepath = os.path.join(models_dir, filename)

    if os.path.exists(filepath):
        print(f"✅ {filename} 已存在")
        continue

    print(f"📥 下载 {filename}...")
    try:
        urllib.request.urlretrieve(url, filepath)
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        print(f"✅ {filename} 下载成功 ({file_size:.2f} MB)")
    except Exception as e:
        print(f"❌ {filename} 下载失败: {e}")
        print(f"   请手动下载: {url}")

print("="*60)
print(f"✅ 模型文件保存在: {models_dir}")
print("\n现在可以运行 python app.py 启动服务器了！")
