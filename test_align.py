import cv2
from deepface import DeepFace
import glob
import os
import matplotlib.pyplot as plt

images = glob.glob(r"c:\Users\dsaiv\OneDrive\Desktop\final year project\Latent_Facial_Identity_Mapping\uploads\composite_*.png")

for img_path in images[:1]:
    img = cv2.imread(img_path)
    img_rs = cv2.resize(img, (500, 500))
    
    # Extract with alignment
    faces_aligned = DeepFace.extract_faces(img_rs, detector_backend='opencv', enforce_detection=False, align=True)
    # Extract without alignment
    faces_unaligned = DeepFace.extract_faces(img_rs, detector_backend='opencv', enforce_detection=False, align=False)
    
    cv2.imwrite("aligned_face.jpg", cv2.cvtColor(faces_aligned[0]['face'] * 255, cv2.COLOR_RGB2BGR))
    cv2.imwrite("unaligned_face.jpg", cv2.cvtColor(faces_unaligned[0]['face'] * 255, cv2.COLOR_RGB2BGR))
    print("Saved aligned_face.jpg and unaligned_face.jpg")
