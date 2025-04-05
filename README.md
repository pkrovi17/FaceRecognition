# Face Recognition System Using Haarcascade and OpenCV

This is a simple face recognition project built with Python and OpenCV. It uses the Haarcascade classifier for face detection and the Local Binary Patterns Histograms (LBPH) recognizer for identifying faces.

---

## 📁 Project Structure

- `create_data.py` – Captures and stores 120 face images of a user from different angles using a webcam.
- `face_recognize.py` – Trains the recognizer on the stored images and performs real-time face recognition.
- `haarcascade_frontalface_default.xml` – The pre-trained Haarcascade model used for face detection.
- `datasets/` – Directory where user face data is stored.

---

## 🔧 Setup

### Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy

Install dependencies:

```bash
pip install opencv-python numpy
