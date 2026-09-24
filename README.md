# Handwritten Digit Recognition System (Deep Learning CNN)

An interactive computer vision and deep learning web application powered by a Convolutional Neural Network (CNN) trained on the MNIST benchmark dataset.

---

## 🌐 Live Deployment Links

- **Public Live URL (Active):** [https://eloise-thatchy-cherlyn.ngrok-free.dev](https://eloise-thatchy-cherlyn.ngrok-free.dev)
- **Local Application URL:** [http://127.0.0.1:5000](http://127.0.0.1:5000)

*(Note for first-time visitors to the ngrok URL: Click "Visit Site" or send header `ngrok-skip-browser-warning: 1` if prompted by ngrok's free tier gateway).*

---

## 📊 Model Architecture & Benchmarks

| Model Architecture | Category | Accuracy on MNIST Test Set |
| :--- | :--- | :--- |
| **Convolutional Neural Network (CNN)** | **Deep Learning (Our Model)** | **99.06%** |
| K-Nearest Neighbors (KNN, k=3) | Machine Learning | 97.05% |
| Support Vector Classifier (SVM, RBF) | Machine Learning | 96.95% |

### CNN Pipeline:
- **Input:** 28×28 Grayscale (`float32` [0, 1])
- **Conv2D Layer 1:** 32 filters, 3×3 kernel, ReLU activation
- **MaxPooling2D Layer 1:** 2×2 pool size
- **Conv2D Layer 2:** 64 filters, 3×3 kernel, ReLU activation
- **MaxPooling2D Layer 2:** 2×2 pool size
- **Flatten Layer:** 1,600 features
- **Dense Layer:** 128 hidden neurons, ReLU activation
- **Dropout Layer:** 0.3 rate for regularization
- **Output Layer:** 10 neurons, Softmax activation (Digits 0–9)
- **Image Preprocessing:** Bounding-box crop + Center-of-Mass translation to match MNIST benchmark distribution.

---

## 👥 Project Team

- **Choyon Dhor** — ID: `231-115-094`
- **Puspo Gondha Paul** — ID: `231-115-082`
- **Department:** Computer Science & Engineering, CSE 58th Batch, Section C
- **Institution:** Metropolitan University, Sylhet, Bangladesh

---

## 🚀 Running the Project Locally

### 1. Activate Environment & Run Live Server
```powershell
# Launch Flask server with active public tunnel
.\.venv\Scripts\python.exe run_live.py
```

### 2. Run Local Web App Only
```powershell
.\.venv\Scripts\python.exe app.py
```

---

## ☁️ Permanent Free Cloud Deployment Options

The project includes ready-to-deploy configuration files: [`requirements.txt`](file:///c:/Users/suman/Documents/ALL%20Projects/Hand-written-digit-recognition/requirements.txt), [`Procfile`](file:///c:/Users/suman/Documents/ALL%20Projects/Hand-written-digit-recognition/Procfile), and [`app.py`](file:///c:/Users/suman/Documents/ALL%20Projects/Hand-written-digit-recognition/app.py).

### Option A: Hugging Face Spaces (Recommended for AI/ML)
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Select **Docker** or **Gradio/Flask** (SDK: `Docker` or `Static/Python`).
3. Push this directory (`app.py`, `digit_cnn_model.h5`, `requirements.txt`).
4. Your permanent free link will be live at `https://huggingface.co/spaces/<username>/<space-name>`.

### Option B: Render.com
1. Create a new **Web Service** on [render.com](https://render.com).
2. Connect your Git repository.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app` (or `python app.py`)
