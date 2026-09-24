# Handwritten Digit Recognition System (Deep Learning CNN)

An interactive computer vision and deep learning web application powered by a Convolutional Neural Network (CNN) trained on the MNIST benchmark dataset.

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/ChoyonDhor/hand-written-digit-recognition)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-6.0%2B-orange.svg)](https://gradio.app/)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-99.06%25-brightgreen.svg)]()

---

## 🌐 24/7 Permanent Live Demo

Try the interactive digit canvas directly in your browser (no installation required, runs 24/7 on cloud infrastructure):

👉 **[https://huggingface.co/spaces/ChoyonDhor/hand-written-digit-recognition](https://huggingface.co/spaces/ChoyonDhor/hand-written-digit-recognition)**

---

## 📊 Model Architecture & Benchmarks

| Model Architecture | Category | Accuracy on MNIST Test Set |
| :--- | :--- | :--- |
| **Convolutional Neural Network (CNN)** | **Deep Learning (Our Model)** | **99.06%** |
| K-Nearest Neighbors (KNN, k=3) | Machine Learning Baseline | 97.05% |
| Support Vector Classifier (SVM, RBF) | Machine Learning Baseline | 96.95% |

### CNN Pipeline Specifications:
- **Input Layer:** 28×28 Grayscale (`float32` [0, 1])
- **Conv2D Layer 1:** 32 filters, 3×3 kernel, ReLU activation
- **MaxPooling2D Layer 1:** 2×2 pool size
- **Conv2D Layer 2:** 64 filters, 3×3 kernel, ReLU activation
- **MaxPooling2D Layer 2:** 2×2 pool size
- **Flatten Layer:** 1,600 features
- **Dense Layer:** 128 hidden neurons, ReLU activation
- **Dropout Layer:** 0.3 rate for regularization
- **Output Layer:** 10 neurons, Softmax activation (Digits 0–9)
- **Image Preprocessing:** Bounding-box crop + Aspect-ratio scaling to 20×20 + Center-of-Mass translation to (13.5, 13.5) to match MNIST benchmark distribution.

---

## 👥 Project Team

- **Choyon Dhor** — ID: `231-115-094`
- **Puspo Gondha Paul** — ID: `231-115-082`
- **Department:** Computer Science & Engineering, CSE 58th Batch, Section C
- **Institution:** Metropolitan University, Sylhet, Bangladesh

---

## 🚀 Running the Project Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Gradio Web Application
```bash
python app.py
```
Open your browser at `http://127.0.0.1:7860` to access the interactive drawing canvas.

---

## 📁 Repository Structure

```text
├── app.py                           # Gradio web app & inference pipeline
├── digit_cnn_pytorch.pt             # PyTorch trained model weights (99.06% accuracy)
├── digit_cnn_model.h5               # Original Keras/TensorFlow model weights
├── train.py                         # CNN training script on MNIST
├── Hand_Written_digit_Recognition_cnn.ipynb  # Original research & analysis notebook
├── requirements.txt                 # Project dependencies
├── .gitignore                       # Ignored cache & virtual environment files
└── README.md                        # Documentation & live demo link
```
