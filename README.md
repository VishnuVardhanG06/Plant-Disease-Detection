# 🌿 Plant Disease Detection

> An AI-powered web application for detecting plant leaf diseases using classical machine learning — built with HOG feature extraction, Random Forest, and SVM classifiers.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0%2B-black?logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv)
![License](https://img.shields.io/badge/License-Educational-lightgrey)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Supported Diseases](#-supported-diseases)
- [Dataset Setup](#-dataset-setup)
- [Installation](#-installation)
- [Training the Model](#-training-the-model)
- [Running the Web App](#-running-the-web-app)
- [How to Predict](#-how-to-predict)
- [ML Pipeline](#-ml-pipeline)
- [Technologies Used](#-technologies-used)

---

## 🔍 Overview

This project provides two ways to detect plant leaf diseases:

1. **Web Application** — Upload a leaf image via a modern browser UI and instantly get the disease name, confidence score, severity, and prevention/treatment tips.
2. **CLI Pipeline** — Run the full training + evaluation pipeline from the command line.

**Key features:**
- 🌿 Supports Tomato and Potato disease classes
- 🤖 Two ML models: Random Forest and SVM (HOG features)
- 📊 Automatic model comparison and best model selection
- 💊 Disease tips: symptoms, prevention, and treatment
- 📱 Responsive web UI with drag-and-drop image upload

---

## 📁 Project Structure

```
Disease Detection/
│
├── dataset/                          # Leaf images (one folder per disease class)
│   ├── Tomato___Early_blight/
│   ├── Tomato___Late_blight/
│   ├── Tomato___healthy/
│   ├── Potato___Early_blight/
│   ├── Potato___Late_blight/
│   └── Potato___healthy/
│
├── src/
│   ├── config.py                     # All hyperparameters and paths
│   ├── data_loader.py                # Load images and labels
│   ├── preprocess.py                 # Resize, grayscale, normalize
│   ├── feature_extraction.py         # HOG descriptor extraction
│   ├── train_model.py                # Random Forest and SVM training
│   ├── evaluate_model.py             # Metrics, confusion matrix, charts
│   └── predict.py                    # CLI single-image inference
│
├── templates/
│   └── index.html                    # Web app HTML
│
├── static/
│   ├── css/style.css                 # Premium UI styling
│   └── js/app.js                     # Frontend fetch API and animations
│
├── trained_models/
│   └── best_model.pkl                # Saved after training
│
├── reports/                          # Auto-generated charts and outputs
│   ├── confusion_matrix_Random_Forest.png
│   ├── confusion_matrix_Support_Vector_Machine.png
│   ├── model_accuracy_comparison.png
│   └── prediction_result.png
│
├── app.py                            # Flask web server
├── main.py                           # ML training pipeline entry point
├── disease_tips.json                 # Disease info database
├── requirements.txt
└── README.md
```

---

## 🦠 Supported Diseases

| Class Folder | Display Name |
|---|---|
| `Tomato___Early_blight` | Tomato Early Blight |
| `Tomato___Late_blight` | Tomato Late Blight |
| `Tomato___healthy` | Healthy Tomato |
| `Tomato__Target_Spot` | Tomato Target Spot |
| `Potato___Early_blight` | Potato Early Blight |
| `Potato___Late_blight` | Potato Late Blight |
| `Potato___healthy` | Healthy Potato |

> The app automatically detects all sub-folders in [dataset/](cci:1://file:///c:/Users/Other/OneDrive/Desktop/Disease%20Detection/src/data_loader.py:50:0-106:38) as disease classes — add any new class folder and retrain.

---

## 🗂️ Dataset Setup

1. Download the **PlantVillage dataset** from [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease).
2. Extract and copy the class folders into [dataset/](cci:1://file:///c:/Users/Other/OneDrive/Desktop/Disease%20Detection/src/data_loader.py:50:0-106:38):

```
dataset/
    Tomato___Early_blight/
        img001.jpg
        img002.jpg
    Tomato___healthy/
        ...
```

> Each sub-folder name becomes the class label. Supported formats: `.jpg`, `.jpeg`, [.png](cci:7://file:///c:/Users/Other/OneDrive/Desktop/Disease%20Detection/reports/prediction_result.png:0:0-0:0), `.bmp`

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd "Disease Detection"
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Training the Model

Run the full ML pipeline from the project root:

```bash
py main.py
```

### Pipeline steps

| Step | Action |
|---|---|
| 1 | Load all images from [dataset/](cci:1://file:///c:/Users/Other/OneDrive/Desktop/Disease%20Detection/src/data_loader.py:50:0-106:38) |
| 2 | Resize → 128×128, grayscale, normalize |
| 3 | Extract HOG feature vectors |
| 4 | Encode labels and stratified 80/20 split |
| 5 | Train Random Forest (200 trees) |
| 6 | Train SVM (linear kernel) |
| 7 | Evaluate both, save reports to [reports/](cci:1://file:///c:/Users/Other/OneDrive/Desktop/Disease%20Detection/src/evaluate_model.py:33:0-35:43) |
| 8 | Save best model to [trained_models/best_model.pkl](cci:7://file:///c:/Users/Other/OneDrive/Desktop/Disease%20Detection/trained_models/best_model.pkl:0:0-0:0) |

> ⚡ **Tip:** SVM is set to `linear` kernel for fast training. Change `SVM_KERNEL` in [src/config.py](cci:7://file:///c:/Users/Other/OneDrive/Desktop/Disease%20Detection/src/config.py:0:0-0:0) if needed.

---

## 🌐 Running the Web App

Start the Flask server after training:

```bash
py app.py
```

Then open your browser at:

```
http://127.0.0.1:5000
```

### Usage

1. **Upload** a leaf image by dragging and dropping or clicking the upload zone
2. Click **🔍 Analyze Leaf**
3. View results:
   - Disease name and severity badge
   - Animated confidence percentage ring
   - Symptoms, Prevention, and Treatment tabs

> **Important:** Always restart `py app.py` after retraining so the server loads the new model.

---

## 🔍 How to Predict (CLI)

```bash
py src/predict.py path/to/leaf_image.jpg
```

Optional — specify a custom model path:

```bash
py src/predict.py path/to/leaf_image.jpg --model trained_models/best_model.pkl
```

The result image is saved to [reports/prediction_result.png](cci:7://file:///c:/Users/Other/OneDrive/Desktop/Disease%20Detection/reports/prediction_result.png:0:0-0:0).

---

## 🧠 ML Pipeline

```
Raw Images (any resolution)
       │
       ▼
Preprocessing — resize 128×128 · grayscale · normalize [0,1]
       │
       ▼
HOG Feature Extraction — 9 orientations · 8×8 cells · L2-Hys norm
       │
       ▼
Stratified 80/20 Train / Test Split
       │
       ├──► Random Forest (200 trees, all CPU cores)
       │
       └──► SVM (linear kernel, C=10)
                    │
                    ▼
           Evaluation (Accuracy · F1 · Confusion Matrix)
                    │
                    ▼
         Best Model → trained_models/best_model.pkl
```

---

## 📊 Evaluation Outputs

All generated automatically in [reports/](cci:1://file:///c:/Users/Other/OneDrive/Desktop/Disease%20Detection/src/evaluate_model.py:33:0-35:43):

| File | Description |
|---|---|
| [confusion_matrix_Random_Forest.png](cci:7://file:///c:/Users/Other/OneDrive/Desktop/Disease%20Detection/reports/confusion_matrix_Random_Forest.png:0:0-0:0) | Confusion matrix for RF |
| [confusion_matrix_Support_Vector_Machine.png](cci:7://file:///c:/Users/Other/OneDrive/Desktop/Disease%20Detection/reports/confusion_matrix_Support_Vector_Machine.png:0:0-0:0) | Confusion matrix for SVM |
| [model_accuracy_comparison.png](cci:7://file:///c:/Users/Other/OneDrive/Desktop/Disease%20Detection/reports/model_accuracy_comparison.png:0:0-0:0) | Side-by-side accuracy bar chart |
| [prediction_result.png](cci:7://file:///c:/Users/Other/OneDrive/Desktop/Disease%20Detection/reports/prediction_result.png:0:0-0:0) | Input image with predicted label overlaid |

---

## 🛠️ Technologies Used

| Library | Version | Purpose |
|---|---|---|
| `Flask` | ≥ 3.0 | Web server and API |
| `opencv-python` | ≥ 4.8 | Image I/O and colour conversion |
| `scikit-image` | ≥ 0.21 | HOG feature descriptor |
| `scikit-learn` | ≥ 1.3 | Random Forest, SVM, metrics |
| `numpy` | ≥ 1.24 | Vectorised array operations |
| `matplotlib` | ≥ 3.7 | Chart and confusion matrix visualisation |
| `tqdm` | ≥ 4.65 | Feature extraction progress bar |

---

## 📝 Notes

- **No GPU required** — runs entirely on a standard laptop CPU
- All hyperparameters are in [src/config.py](cci:7://file:///c:/Users/Other/OneDrive/Desktop/Disease%20Detection/src/config.py:0:0-0:0) — change once, applies everywhere
- Adding new disease classes: create a new folder in [dataset/](cci:1://file:///c:/Users/Other/OneDrive/Desktop/Disease%20Detection/src/data_loader.py:50:0-106:38), retrain with `py main.py`, restart `py app.py`

---

## 📖 License

Created for educational purposes as part of a machine learning project.
