# ♻️ Swachh Vision AI – Waste Detection using YOLOv5

An End-to-End Computer Vision project for **automatic waste detection and classification** using YOLOv5.
This system can detect waste objects from images and can be integrated into smart city / cleanliness monitoring solutions.

---

## 🚀 Project Workflow

The project follows a modular pipeline architecture:

1. **Constants** → Stores global configuration values
2. **Entity** → Defines structured data objects
3. **Components** → Data ingestion, validation, training modules
4. **Pipelines** → Training and prediction pipelines
5. **Flask App (`app.py`)** → User interface for prediction

---

## 📂 Project Structure

```
.
├── app.py
├── requirements.txt
├── wasteDetection/
├── templates/
├── data/
├── yolov5/
└── Dockerfile
```

---

## 🧠 Model

* Model Used → **YOLOv5**
* Task → Waste Object Detection
* Output → Bounding boxes + Class labels

---

## ⚙️ How to Run Locally

### Step 1 — Clone Repository

```
git clone https://github.com/saumyasrivastava21/SWACHH-VISION-AI.git
cd SWACHH-VISION-AI
```

---

### Step 2 — Create Virtual Environment

```
conda create -n waste python=3.7 -y
conda activate waste
```

---

### Step 3 — Install Requirements

```
pip install -r requirements.txt
```

---

### Step 4 — Run Application

```
python app.py
```

Now open your browser and go to:

```
http://localhost:5000
```

Upload an image → Get waste detection prediction.

---

## 📸 Sample Use Cases

✅ Smart City Monitoring
✅ Waste Segregation Systems
✅ Environmental AI Solutions
✅ Industrial Garbage Detection

---

## 👨‍💻 Author

**Saumya Srivastava**
B.Tech IT | Java Full Stack + AI/ML

---
