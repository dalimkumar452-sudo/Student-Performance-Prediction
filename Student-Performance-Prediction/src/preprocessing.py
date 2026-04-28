import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_data(n=1000):
    """Generates synthetic student data for simulation."""
    np.random.seed(42)
    data = {
        'study_hours': np.random.uniform(1, 15, n),
        'attendance_pct': np.random.uniform(50, 100, n),
        'quiz_avg': np.random.uniform(30, 100, n),
        'assignment_score': np.random.uniform(40, 100, n),
        'lms_engagement': np.random.uniform(5, 60, n),
        'prior_gpa': np.random.uniform(2.0, 4.0, n)
    }
    df = pd.DataFrame(data)
    
    # Target Logic: 1 = At Risk, 0 = Safe
    # Calculated based on weighted importance of features
    score = (df['attendance_pct'] * 0.35 + df['quiz_avg'] * 0.25 + 
             df['study_hours'] * 2.5 + df['assignment_score'] * 0.2)
    df['at_risk'] = (score < 55).astype(int) 
    return df

def preprocess(df):
    """Splits and scales the data."""
    X = df.drop('at_risk', axis=1)
    y = df['at_risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns