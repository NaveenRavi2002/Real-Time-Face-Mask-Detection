# Face Mask Detection App 😷

A real-time face mask detection application using Convolutional Neural Network (CNN) and Streamlit web interface.

## ✨ Features

- 🎯 Real-time face detection using OpenCV Haar Cascade
- 🧠 Custom CNN model for mask classification
- 🎨 Color-coded bounding boxes (Green = Mask, Blue = No Mask)
- 🌐 Interactive web interface with Streamlit
- 📊 Data augmentation for better model accuracy

## 🧱 Project Structure

```
.
├── model.py                # Training script for CNN model
├── app.py                  # Streamlit web application
├── mask_detector.h5        # Trained model (generated after training)
├── Train/                  # Training dataset
│   ├── WithMask/          # Images with face masks
│   └── WithoutMask/       # Images without face masks
├── Test/                   # Testing dataset
│   ├── WithMask/
│   └── WithoutMask/
└── README.md
```

## 📂 Dataset Setup

Create the following folder structure and add your images:

```
Train/
    WithMask/
        mask_001.jpg
        mask_002.jpg
        ...
    WithoutMask/
        no_mask_001.jpg
        no_mask_002.jpg
        ...

Test/
    WithMask/
        test_mask_001.jpg
        ...
    WithoutMask/
        test_no_mask_001.jpg
        ...
```

**Note:** You need at least 100+ images in each category for decent results.

## 🛠 Installation

### Prerequisites
- Python 3.8 or higher
- Webcam (for real-time detection)

### Install Dependencies

```bash
pip install tensorflow opencv-python streamlit pillow numpy
```

Or create a `requirements.txt`:

```txt
tensorflow==2.15.0
opencv-python==4.8.1.78
streamlit==1.29.0
pillow==10.1.0
numpy==1.24.3
```

Then install:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Step 1: Train the Model

1. Prepare your dataset in `Train/` and `Test/` folders
2. Run the training script:

```bash
python model.py
```

**Training details:**
- Image size: 64×64 pixels
- Epochs: 20
- Batch size: 32
- Optimizer: Adam
- Data augmentation: Shear, Zoom, Horizontal Flip

After training, you'll see:
```
Model saved as mask_detector.h5
```

### Step 2: Run the Application

```bash
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

### Step 3: Use the App

1. Click **"Open Camera"** button
2. Allow browser to access your webcam
3. The app will detect faces and show predictions:
   - **Green box + "Mask"** = Face mask detected ✅
   - **Red box + "No Mask"** = No face mask detected ❌
4. Click **"Close Camera"** to stop

## 🧠 Model Architecture

```
Input (64x64x3)
    ↓
Conv2D (32 filters) + ReLU → MaxPooling
    ↓
Conv2D (64 filters) + ReLU → MaxPooling
    ↓
Conv2D (128 filters) + ReLU → MaxPooling
    ↓
Flatten
    ↓
Dense (128) + ReLU
    ↓
Dropout (0.5)
    ↓
Dense (1) + Sigmoid → Output (0 or 1)
```

## 😷 How It Works

1. **Face Detection:** OpenCV Haar Cascade detects faces in the video frame
2. **Preprocessing:** Each face is resized to 64×64 and normalized (0-1)
3. **Prediction:** CNN model predicts mask probability
   - Output < 0.5 → **Mask**
   - Output ≥ 0.5 → **No Mask**
4. **Visualization:** Bounding box and label drawn on the frame

## 🧪 Troubleshooting

### Model Not Found
```
OSError: Unable to open file 'mask_detector.h5'
```
**Solution:** Run `python model.py` first to train and save the model.

### Dataset Error
```
Found 0 images belonging to 0 classes
```
**Solution:** 
- Verify `Train/` and `Test/` folders exist
- Ensure images are in `WithMask/` and `WithoutMask/` subfolders
- Check image formats (jpg, jpeg, png)

### Webcam Issues
```
Failed to grab frame
```
**Solutions:**
- Check camera permissions in OS settings
- Close other apps using the webcam (Zoom, Skype, etc.)
- Try different camera index:
  ```python
  cam = cv2.VideoCapture(1)  # or 2, 3, etc.
  ```

### Haar Cascade Error
```
Can't open file: 'haarcascade_frontalface_default.xml'
```
**Solution:** The code already uses the correct built-in path. If error persists:
```python
print(cv2.data.haarcascades)  # Check the path
```

### Low Accuracy
**Solutions:**
- Add more diverse training images (different angles, lighting, backgrounds)
- Increase epochs: Change `epochs=20` to `epochs=30` or `50`
- Use larger image size: Change `target_size=(64, 64)` to `(128, 128)`
- Balance dataset: Equal number of mask/no-mask images

## 📊 Improve Model Performance

```python
# In model.py, modify these parameters:

# More epochs
model.fit(..., epochs=50)

# Larger image size
target_size=(128, 128)
input_shape=(128, 128, 3)

# Lower learning rate
optimizer=Adam(learning_rate=0.0001)

# More augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,
    brightness_range=[0.8, 1.2]
)
```

## 🌐 Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add `mask_detector.h5` to your repo

**For deployment, use:**
```txt
tensorflow==2.15.0
opencv-python-headless==4.8.1.78
streamlit==1.29.0
pillow==10.1.0
numpy==1.24.3
```

Note: Use `opencv-python-headless` for cloud deployment.

## 📸 Screenshots

**Training Output:**
```
Found 800 images belonging to 2 classes.
Found 200 images belonging to 2 classes.
Epoch 1/20
25/25 [==============================] - 15s 580ms/step
...
Epoch 20/20
25/25 [==============================] - 12s 490ms/step
Model saved as mask_detector.h5
```

**App Interface:**
- Webcam feed with real-time detection
- Green/Red bounding boxes
- "Mask" or "No Mask" labels

## 📝 Tips

- **Better lighting** improves face detection accuracy
- **Front-facing** poses work best
- **Clear images** in training data = better results
- **Balanced dataset** prevents bias (50% mask, 50% no mask)

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add more features
- Improve model architecture
- Enhance UI/UX
- Fix bugs

## 📄 License

This project is open-source and available for educational purposes.

---

**Made with ❤️ using TensorFlow, OpenCV & Streamlit**
