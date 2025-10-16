import pandas as pd
import joblib
from typing import Dict
from src.azgems.repositories.get_dataset import DatasetPreparation
from src.azgems.Services.OpenAIclient import OpenAIClient
from collections import OrderedDict


class ModelInference:
    def __init__(
        self, customer_name: str, start_timestamp, end_timestamp, po_comitted: str
    ):
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
        - sorted feature importance (descending)
        """
        # 1️⃣ Preprocess input
        df_scaled = self.preprocess_input(data_point)

        # 2️⃣ Prediction
        prediction = self.model.predict(df_scaled)[0]
        probability = self.model.predict_proba(df_scaled)[0]

        # 3️⃣ Feature importance (sorted descending)
        feature_importance = dict(
            zip(df_scaled.columns, self.model.feature_importances_)
        )
        feature_importance_sorted = OrderedDict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

        return prediction, probability, feature_importance_sorted

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
        Write the reason why the shipment may have been potentially delayed in a single line considering the feature importance that is provided.
        Instead of just relaying on the feature importance value alone, try to come up with why those features may have contributed to the delay.
        Use Simple language and avoid using too many words.
        Here are detail of the features columns that are used for training the model:
        - quantity_in: The quantity of the shipment
        - seal: The seal of the shipment
        - yield_percentage: The yield percentage of the shrimps
        - gap_between_due_and_eta: The gap between the due date the bill must be paid and the eta (estimated time of arrival of shipment)
        - gap_between_shipped_and_due: The gap between the shipped date(date when shipment is marked as shipped) and the due date(the date when the bill must be paid)
        - gap_between_bill_and_shipment: The gap between the bill date(the date when the bill was issued) and the shipped date(the date when shipment is marked as shipped)
        - gap_between_eta_shipped: The gap between the eta date(estimated time of arrival of shipment) and the shipped date(the date when shipment is marked as shipped)
        - gap_between_eta_bill: The gap between the eta date(estimated time of arrival of shipment) and the bill date(the date when the bill was issued)
        - due_gap: The gap between the due date(the date when the bill must be paid) and the bill date(the date when the bill was issued)
        - responsiveness: The responsiveness of the shipment (the gap between the bill date and the shipped date)

        Flow of the shipment:
        first bill is issued along with due date, then shipment is marked as shipped, then eta is estimated, then shipment is received.
        
        Example:
        Delayed may be due to the following reasons:
        - The gap between the due date and the eta is too long
        - The gap between the shipped date and the due date is too long
        - The gap between the bill date and the shipped date is too long
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
            print(feature_importance)
            print(f"Probability: {probability}")
            print(f"Feature importance: {feature_importance}")
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
                        "batch_in_id": batch_id,
                        "title": "Shipment may be delayed for batch number: "
                        + batch_number,
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
    model_loader = ModelInference(
        customer_name="Walmart",
        start_timestamp="2025-10-13",
        end_timestamp="2025-10-15",
        po_comitted="Direct Sale",
    )

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
