# Latent-space Facial Identity Mapping

Project scaffold for Latent-space Facial Identity Mapping.

See `requirements.txt` for Python dependencies and `app.py` to run the demo Flask server.

Quick start:

1. Create a Python 3.8+ virtual environment.
2. Install dependencies:

   pip install -r requirements.txt

Note: `face_recognition` requires `dlib` which may need a C++ build toolchain on Windows. If you have trouble installing, consider using Windows Subsystem for Linux (WSL) or a prebuilt wheel for `dlib`.

3. Add dataset images into `dataset/` with filenames like `person1.jpg`, `person2.jpg`.
4. Populate the database encodings:

   python database/add_faces.py --dataset ../dataset --db ../database/criminals.db

5. Run the app:

   python app.py

Then open http://127.0.0.1:5000 in your browser and upload a sketch to match against the dataset.