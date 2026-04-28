import os
import joblib
import uvicorn
import webbrowser
import pandas as pd
import plotly.express as px
from xgboost import XGBClassifier
from fastapi import FastAPI
from sklearn.metrics import classification_report

# Internal Module Imports
from src.preprocessing import get_data, preprocess

# Initialize Directory Structure
for folder in ['models', 'outputs', 'images']:
    os.makedirs(folder, exist_ok=True)

app = FastAPI(title="Student Performance Prediction API")

# Global variables for model and scaler
df = get_data()
X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(df)

# Initialize XGBoost Model
model = XGBClassifier(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=5, 
    random_state=42
)

def run_training():
    """Trains the XGBoost model and saves artifacts."""
    print("🚀 Training Advanced XGBoost Model...")
    model.fit(X_train, y_train)
    
    # Persist model and scaler
    joblib.dump(model, 'models/xgboost_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    y_pred = model.predict(X_test)
    print("✅ Training Complete. Evaluation report saved to 'outputs/metrics.txt'")
    
    # Save metrics with UTF-8 encoding
    with open('outputs/metrics.txt', 'w', encoding='utf-8') as f:
        f.write(classification_report(y_test, y_pred))

def open_dashboard():
    """Generates visual analytics, saves them as images, and creates HTML dashboard."""
    print("📊 Generating Visual Analytics & Saving Images...")
    
    # 1. Feature Importance Plot
    importance_df = pd.DataFrame({
        'Feature': feature_names, 
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=True)
    
    fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h', 
                     title="Key Risk Factor Importance", color='Importance')
    
    # --- AUTO SAVE IMAGE 1 ---
    fig_imp.write_image("images/feature_importance.png")
    
    # 2. Risk Distribution Pie Chart
    fig_pie = px.pie(df, names='at_risk', title="Overall Student Risk Status (0: Safe, 1: At Risk)", 
                     hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
    
    # --- AUTO SAVE IMAGE 2 ---
    fig_pie.write_image("images/risk_distribution.png")
    
    # 3. Save to HTML Dashboard
    html_path = os.path.abspath('outputs/dashboard.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write("<html><head><title>Admin Dashboard</title>")
        f.write("<style>body{font-family:sans-serif; background-color:#f4f7f6; padding:40px; text-align:center;}</style></head><body>")
        f.write("<h1>🎓 Student Performance Prediction Dashboard</h1>")
        f.write("<div style='display:block;'>")
        f.write(fig_imp.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fig_pie.to_html(full_html=False, include_plotlyjs=False))
        f.write("</div></body></html>")
    
    print("✅ Plots and Images saved successfully.")
    
    # Auto-open browser
    webbrowser.open('file://' + html_path)

# --- FastAPI Endpoints ---

@app.get("/")
def home():
    return {"message": "Student Performance Prediction API is Online", "version": "v1.0"}

@app.post("/predict")
def predict(study_hours: float, attendance: float, quiz_score: float, assignments: float, lms_log: float, gpa: float):
    """Endpoint to predict risk for a single student."""
    input_data = [[study_hours, attendance, quiz_score, assignments, lms_log, gpa]]
    
    # Use DataFrame to maintain feature names consistency
    input_df = pd.DataFrame(input_data, columns=feature_names)
    scaled_input = scaler.transform(input_df)
    
    prob = model.predict_proba(scaled_input)[0][1]
    prediction = int(model.predict(scaled_input)[0])
    
    return {
        "risk_probability": round(float(prob), 4),
        "prediction": "At Risk" if prediction == 1 else "Safe",
        "action_required": bool(prediction)
    }

# --- Execution Entry Point ---
if __name__ == "__main__":
    # 1. Run Pipeline & Visuals
    run_training()
    open_dashboard()
    
    # 2. Launch FastAPI Server
    print("🌐 Launching API Server...")
    print("💡 Documentation: http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)