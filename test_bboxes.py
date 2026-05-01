import cv2
import numpy as np
from deepface import DeepFace
import glob
import os

images = glob.glob(r"c:\Users\dsaiv\OneDrive\Desktop\final year project\Latent_Facial_Identity_Mapping\**\*.png", recursive=True)

for img_path in images[:3]:
    print(f"\n--- Testing {os.path.basename(img_path)} ---")
    img_bgr = cv2.imread(img_path)
    img_bgr = cv2.resize(img_bgr, (500, 500))
    
    for backend in ["opencv", "mtcnn", "retinaface"]:
        try:
            res = DeepFace.extract_faces(img_bgr, detector_backend=backend, enforce_detection=True)
            if res:
                facial_area = res[0]["facial_area"]
                print(f"{backend:10s} SUCCESS. BBox: {facial_area}")
        except Exception as e:
            print(f"{backend:10s} FAILED: {str(e)[:50]}")
