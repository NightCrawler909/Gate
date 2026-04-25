import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class AnomalyDetector:
    def __init__(self, model_path='model.pkl', scaler_path='scaler.pkl'):
        # Store model/scaler paths
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        
    def train_model(self, csv_path='custom_dataset.csv'):
        # Load synthetic data
        df = pd.read_csv(csv_path)
        feature_cols = [
            'speed_variance', 'distance_to_restricted_zone', 
            'time_near_boundary', 'time_inside_zone', 
            'sudden_stops', 'carrying_baggage', 'time_of_day_multiplier'
        ]
        
        X = df[feature_cols]
        y = df['intent_class']
        
        # Scale
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train RF
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)
        
        # Save artifacts
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)

    def predict_intent(self, features_dict):
        # Load artifacts if unloaded
        if self.model is None or self.scaler is None:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
            else:
                return 'unknown' # or raise Error
                
        feature_cols = [
            'speed_variance', 'distance_to_restricted_zone', 
            'time_near_boundary', 'time_inside_zone', 
            'sudden_stops', 'carrying_baggage', 'time_of_day_multiplier'
        ]
        
        # Parse inference vector
        x_input = np.array([[features_dict.get(c, 0.0) for c in feature_cols]])
        
        # Predict
        x_scaled = self.scaler.transform(x_input)
        return self.model.predict(x_scaled)[0]
