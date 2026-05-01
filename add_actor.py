import os
import sys

# Ensure imports work from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.feature_extractor import get_face_encoding_from_image
from utils.database import add_criminal

db_path = os.path.join("database", "criminals.db")
img_rel_path = "Vijay_Deverakonda/Vijay_Devarakonda.jpg"
img_abs_path = os.path.join("dataset", img_rel_path)

if not os.path.exists(img_abs_path):
    print(f"Error: {img_abs_path} does not exist.")
    sys.exit(1)

print(f"Extracting features from {img_abs_path}...")
enc = get_face_encoding_from_image(img_abs_path)

if enc is None:
    print("No face detected in the image.")
    sys.exit(1)

print("Adding to database...")
# Note: the app relies on the filepath storing its route inside the dataset
add_criminal(db_path, "Vijay Deverakonda", img_rel_path, enc)

print("Successfully added Vijay Deverakonda to the database!")
