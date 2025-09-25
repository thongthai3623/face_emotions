import cv2
import numpy as np
from tensorflow.keras.models import load_model
import config

# Tải mô hình và nhãn
_model = None
_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

def load_emotion_model():
    global _model
    if _model is None:
        _model = load_model(config.model_path)
    return _model, _labels

def predict_image(img_bgr):
    # img_bgr: ảnh màu BGR từ OpenCV
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    ).detectMultiScale(gray, 1.3, 5)

    results = []
    model, labels = load_emotion_model()
    if len(faces) == 0:
        return [{'emotion': 'neutral', 'confidence': 0.0, 'box': None}]
    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48,48)).astype('float32')/255.0
        face = face.reshape(1,48,48,1)
        probs = model.predict(face)[0]
        idx = np.argmax(probs)
        results.append({
            'box': (x,y,w,h),
            'emotion': labels[idx],
            'confidence': float(probs[idx])
        })
    return results