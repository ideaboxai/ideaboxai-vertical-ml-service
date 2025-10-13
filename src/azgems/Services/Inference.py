import pandas as pd
import joblib
from typing import Dict


class ModelInference:
    def __init__(self, model_path: str):
        """
        Load the saved model, scaler, encoders, and feature list.
        """
        saved = joblib.load(model_path)
        self.model = saved["model"]
        self.scaler = saved["scaler"]
        self.encoders = saved["encoders"]
        self.features = saved["features"]

    def preprocess_input(self, data_point: Dict) -> pd.DataFrame:
        """
        Convert a single data point dict into a preprocessed DataFrame
        ready for model prediction.
        """
        # Convert to DataFrame
        df = pd.DataFrame([data_point], columns=self.features)

        for col, le in self.encoders.items():
            if col in df.columns:
                try:
                    df[col] = le.transform(df[col].astype(str))
                except ValueError as e:
                    print(f"\n❌ Encoding error in column: {col}")
                    print(f"   Values in data: {df[col].unique()}")
                    print(f"   Known classes: {le.classes_}")
                    raise e

        # Scale numerical columns
        df_scaled = pd.DataFrame(self.scaler.transform(df), columns=self.features)
        return df_scaled

    def predict(self, data_point: Dict):
        """
        Make prediction and return class label and probability.
        """
        df_scaled = self.preprocess_input(data_point)
        prediction = self.model.predict(df_scaled)[0]
        probability = self.model.predict_proba(df_scaled)[0]
        return prediction, probability


if __name__ == "__main__":
    # Example usage
    model_loader = ModelInference("models/Walmart/logistic_regression.pkl")

    # Dummy data point (use actual feature names from your dataset)
    dummy_data = {
        "quantity_in": 100,
        "bill_date_year": "2025",
        "bill_date_month": "10",
        "bill_date_day": "13",
        "bill_date_weekday": "0",
        "due_date_year": "2025",
        "due_date_month": "10",
        "due_date_day": "15",
        "due_date_weekday": "2",
        "eta_year": "2025",
        "eta_month": "10",
        "eta_day": "14",
        "eta_weekday": "1",
        "yield_percentage": 85.0,
        "seal": "M0676831",
        "customs_broker": "unknown",
        "ocean_freight": 500.0,
        # Add any other features present in self.features
    }

    pred_class, pred_prob = model_loader.predict(dummy_data)
    print("Predicted class:", pred_class)
    print("Class probabilities:", pred_prob)
