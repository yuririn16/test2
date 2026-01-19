import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image
import cv2

# MediaPipeの顔検出機能をセットアップ
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

st.title("Android顔認識カメラアプリ")
st.write("Qiitaの記事を参考に、MediaPipeを組み込みました")

# カメラ入力（スマホのカメラが起動します）
img_file = st.camera_input("写真を撮る")

if img_file is not None:
    # 1. 撮った画像を読み込む
    image = Image.open(img_file)
    # 2. MediaPipeで処理するために、numpy形式の配列に変換
    image_np = np.array(image)

    # 3. 顔検出を実行
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        # BGRからRGBに変換（MediaPipe用）
        results = face_detection.process(image_np)

        # 4. 検出した顔に枠を描く
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image_np, detection)
            
            st.image(image_np, caption="認識結果", use_container_width=True)
            st.success(f"{len(results.detections)} 人の顔を検出しました！")
        else:
            st.warning("顔が見つかりませんでした。")
