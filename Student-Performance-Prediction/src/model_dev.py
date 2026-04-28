import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

class ModelDevelopment:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=5):
        """Initializes the XGBoost classifier with specified parameters."""
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )

    def train(self, X_train, y_train):
        """Trains the model on provided training data."""
        print("💡 [model_dev] Training model in progress...")
        self.model.fit(X_train, y_train)
        return self.model

    def evaluate(self, X_test, y_test):
        """Evaluates the model and returns a classification report."""
        predictions = self.model.predict(X_test)
        report = classification_report(y_test, predictions)
        accuracy = accuracy_score(y_test, predictions)
        return report, accuracy

    def save_model(self, model_path, scaler, scaler_path):
        """Saves both the model and the scaler to disk."""
        joblib.dump(self.model, model_path)
        joblib.dump(scaler, scaler_path)
        print(f"✅ Model saved to {model_path}")
        print(f"✅ Scaler saved to {scaler_path}")