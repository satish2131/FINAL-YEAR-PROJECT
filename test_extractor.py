import cv2
from model.feature_extractor import get_face_encoding_from_image
import glob
import os

images = glob.glob(r"c:\Users\dsaiv\OneDrive\Desktop\final year project\Latent_Facial_Identity_Mapping\**\*.png", recursive=True)
print("Found PNG images:", len(images))

for img_path in images[:5]:
    encoding = get_face_encoding_from_image(img_path)
    if encoding is not None:
        print(f"Success for {os.path.basename(img_path)}: length {len(encoding)}")
    else:
        print(f"Failed for {os.path.basename(img_path)}")
