# Real-Time Face Mask Detection using CNN & OpenCV

This project detects face masks in real time using a Convolutional Neural Network (CNN) and OpenCV.

It uses:
- A custom CNN model trained in TensorFlow/Keras
- OpenCV Haar Cascade for face detection
- Webcam feed for real-time predictions

## 🧱 Project Structure

```
.
├── mask_detection_training.ipynb   # Jupyter Notebook for training the CNN model
├── realtime_mask_detection.py      # Python script for real-time mask detection with webcam
├── mask_detector_model.keras       # Trained model (saved after training)
├── dataset/                        # Dataset folder (you create this)
│   ├── with_mask/                  # Images of people wearing masks
│   └── without_mask/               # Images of people without masks
└── README.md
```

## 📂 Dataset Structure

Create a folder named `dataset` in the project directory with the following structure:

```
dataset/
    with_mask/
        img1.jpg
        img2.jpg
        ...
    without_mask/
        img10.jpg
        img11.jpg
        ...
```

- `with_mask/` → images of faces wearing masks
- `without_mask/` → images of faces not wearing masks

The notebook automatically creates training and validation splits using `ImageDataGenerator` with `validation_split=0.2`.

## 🛠 Requirements

Install the required libraries:

```bash
pip install tensorflow opencv-python matplotlib scikit-learn numpy
```

**Note:**
- Use Python 3.8+ (recommended)
- If you have a GPU and proper drivers/CUDA installed, TensorFlow can use it automatically.

## 🧠 Model Training (Jupyter Notebook)

**File:** `mask_detection_training.ipynb`

### Steps:

1. **Open the notebook:**
   ```bash
   jupyter notebook mask_detection_training.ipynb
   ```

2. **Check this variable in the notebook and update if needed:**
   ```python
   DATASET_DIR = "dataset"
   ```

3. **Run all cells in order:**
   
   It will:
   - Load images from `dataset/`
   - Create training and validation generators
   - Build a CNN model with:
     - 3× Conv2D + MaxPooling2D + BatchNorm blocks
     - Dense(128) + Dropout(0.5)
     - Output: Dense(1, activation='sigmoid') for binary classification
   - Train the model for a certain number of epochs
   - Save the model as:
     - `mask_detector_model.keras` (best validation accuracy – via ModelCheckpoint)
     - `mask_detector_model_final.keras` (last epoch)

4. **After successful training, make sure you see:**
   ```
   Models saved as 'mask_detector_model.keras' and 'mask_detector_model_final.keras'
   ```

## 🎥 Real-Time Mask Detection

**File:** `realtime_mask_detection.py`

This script:
- Loads the trained model (`mask_detector_model.keras`)
- Uses Haar Cascade for face detection
- Opens the webcam and detects:
  - **MASK** → Green box
  - **NO MASK** → Red box

### Run the Script

Make sure the file `mask_detector_model.keras` is in the same folder as `realtime_mask_detection.py`.

Run:
```bash
python realtime_mask_detection.py
```

A window will open showing:
- Your webcam feed
- Detected faces with bounding boxes and labels:
  - `MASK: 0.xx`
  - `NO MASK: 0.xx`

**Press `q` to quit.**

## 😷 How It Works (Logic)

### Face Detection

Uses OpenCV Haar Cascade:

```python
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
```

This avoids the common error:
```
Can't open file: 'haarcascade_frontalface_default.xml'
```

### Preprocessing

Each face ROI is:
- Cropped from the frame
- Resized to (128, 128)
- Scaled to [0, 1]
- Expanded to shape (1, 128, 128, 3) for the CNN

### Prediction

The model outputs a single probability (sigmoid):

```python
preds = model.predict(face_input)[0][0]
```

- If `preds < 0.5` → **MASK**
- If `preds ≥ 0.5` → **NO MASK**

### Display

Draws rectangle and text:

```python
cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
cv2.putText(frame, label_text, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
```

## 🧪 Troubleshooting

### 1. Haar Cascade Error

**Error:**
```
ERROR:0@... persistence.cpp:531 cv::FileStorage::Impl::open
Can't open file: 'haarcascade_frontalface_default.xml' in read mode
[ERROR] Could not load Haar cascade. Check the XML path.
```

**Fix:**

In `realtime_mask_detection.py`, we use:

```python
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
```

This automatically uses OpenCV's internal haarcascade folder.

If you still get an error, check:
```python
print(cv2.data.haarcascades)
```
to see the actual path being used.

### 2. Webcam Not Opening

If you see:
```
[ERROR] Could not open webcam.
```

**Try:**
- Check your camera permissions (Windows/Ubuntu/macOS privacy settings).
- Change camera index:
  ```python
  cap = cv2.VideoCapture(1)  # or 2, etc.
  ```
- Close any other apps using the webcam.

### 3. Model Not Found

If you get:
```
OSError: No file or directory found at 'mask_detector_model.keras'
```

**Make sure:**
- You ran the training notebook and it saved `mask_detector_model.keras`.
- The `.py` script and the model file are in the same directory.
- Or update this line:
  ```python
  MODEL_PATH = "path/to/mask_detector_model.keras"