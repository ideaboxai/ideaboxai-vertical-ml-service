import pandas as pd
import joblib
from typing import Dict
from src.azgems.repositories.get_dataset import DatasetPreparation
from src.azgems.Services.OpenAIclient import OpenAIClient


class ModelInference:
    def __init__(self, customer_name: str, start_timestamp, end_timestamp, po_comitted: str):
        """
        Load the saved model, scaler, encoders, and feature list.
        """
        saved = joblib.load(f"models/azgems/{customer_name}/model.pkl")
        self.model = saved["model"]
        self.scaler = saved["scaler"]
        self.encoders = saved["encoders"]
        self.features = saved["features"]
        self.customer_name = customer_name
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.po_comitted = po_comitted
        self.llm = OpenAIClient()

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
        Make prediction and return:
        - class label
        - probability
        - feature importance
        """
        # 1️⃣ Preprocess input
        df_scaled = self.preprocess_input(data_point)

        # 2️⃣ Prediction
        prediction = self.model.predict(df_scaled)[0]
        probability = self.model.predict_proba(df_scaled)[0]

        # 3️⃣ Feature importance
        feature_importance = dict(
            zip(df_scaled.columns, self.model.feature_importances_)
        )

        return prediction, probability, feature_importance

    async def get_data_for_inference_from_cube(self):
        """
        Fetch data, run inference row-by-row,
        and return a list of dictionaries containing
        batch_in_id and prediction.
        """
        # 1️⃣ Load data
        get_data = DatasetPreparation(
            customer_name=self.customer_name,
            start_timestamp=self.start_timestamp,
            end_timestamp=self.end_timestamp,
            po_comitted=self.po_comitted,
        ).calculate_target_variable_from_clean_dataset(task="inference")

        system_prompt = """
        You are a helpful assistant that will help analyze the shipment issues and get the reason behind the shipment delay in a single line.
        Just write the reason why the shipment is delayed might have been delayed in a single line.
        Use Simple language and avoid using too many words.
        
        Example:
        Delayed may be 
        """
        user_prompt = """
        Here is the data:
        {data_point}
        Here is the prediction:
        {prediction}
        Here is the probability for each class:
        {probability}
        Here is the feature importance:
        {feature_importance}
        """

        results = []

        for _, row in get_data.iterrows():
            data_point = row.to_dict()
            batch_id = data_point.get("batch_in_id")
            batch_number = data_point.get("batch_number")

            if batch_id is None:
                continue  # skip if batch_in_id missing

            prediction, probability, feature_importance = self.predict(data_point)
            if prediction == "delayed":
                status_message = await self.llm.generate_response(
                    system_prompt,
                    user_prompt.format(
                        data_point=data_point,
                        prediction=prediction,
                        probability=probability,
                        feature_importance=feature_importance,
                    ),
                )

                results.append(
                    {
                        "title": "Shipment may be delayed for batch number: " + batch_number,
                        "prediction": prediction,
                        "message": status_message,
                        "customer_name": self.customer_name,
                        "vendor_id": data_point.get("vendor_id"),
                        "customer_po": data_point.get("purchase_order"),
                        "sku": data_point.get("sku"),
                        "po_comitted": self.po_comitted,
                    }
                )

        return results


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
