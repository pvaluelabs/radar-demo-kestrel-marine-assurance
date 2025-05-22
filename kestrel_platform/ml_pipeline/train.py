import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import pickle

class HullRiskTrainingPipeline:
    """
    ML training pipeline to model marine risk factors.
    Predicts the target variables: hull damage likelihood and premium loading factor.
    """
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.model = None

    def load_and_preprocess(self) -> tuple:
        df = pd.read_csv(self.data_path)
        
        # Features and target variables definition
        features = ["vessel_age", "deadweight_tonnage", "prior_infractions", "crew_cert_rating"]
        
        # TARGET VARIABLE
        target = "target_claim_occurrence"
        
        X = df[features]
        y = df[target]
        
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_risk_model(self):
        X_train, X_test, y_train, y_test = self.load_and_preprocess()
        
        # Fit a robust gradient booster
        self.model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4)
        self.model.fit(X_train, y_train)
        
        preds = self.model.predict(X_test)
        print("ML Training Pipeline Classification Report:")
        print(classification_report(y_test, preds))

    def export_model_binary(self, export_path: str = "hull_risk_gbm.pkl"):
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        with open(export_path, "wb") as f:
            pickle.dump(self.model, f)
