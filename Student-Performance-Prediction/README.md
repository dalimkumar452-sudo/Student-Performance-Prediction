# 🎓 Student Performance Prediction System (Advanced ML Pipeline)

This is an industry-oriented machine learning project designed to predict student performance levels using academic and behavioral signals. It features a modular Python architecture, an automated XGBoost pipeline, and a FastAPI inference service.

## 📊 Project Visuals

### 1. Key Risk Factors
Analysis of features impacting student performance predictions.
![Feature Importance](Student-Performance-Prediction/images/feature_importance.png)

### 2. Risk Distribution
Breakdown of overall student risk status.
![Risk Distribution](Student-Performance-Prediction/images/risk_distribution.png)

---

## 🚀 Key Features
- **Advanced XGBoost Model**: Optimized for high-accuracy classification.
- **Interactive Dashboard**: Automated HTML analytics dashboard with Plotly.
- **FastAPI Integration**: Real-time prediction API for production use.
- **Modular Structure**: Clean separation of preprocessing, training, and serving.

## 🛠️ Tech Stack
- **ML**: XGBoost, Scikit-learn, Pandas, Numpy
- **API/Serving**: FastAPI, Uvicorn
- **Visualization**: Plotly, Kaleido

## 📂 Project Structure
```text
├── data/           # Simulated datasets
├── src/            # Preprocessing & Model Logic
├── models/         # Trained Model & Scaler Artifacts
├── images/         # Auto-generated visualization plots
├── outputs/        # Performance Metrics & Dashboard
└── main.py         # Main Pipeline Orchestrator
⚙️ How to Run
pip install -r requirements.txt

python main.py

Access API Documentation at http://127.0.0.1:8000/docs

Developed by Dalim Kumar 🚀