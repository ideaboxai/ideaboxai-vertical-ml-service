import json
from collections import OrderedDict
from typing import Dict, Tuple, List

import joblib
import numpy as np
import pandas as pd
import shap

from src.azgems.repositories.get_dataset import DatasetPreparation
from src.azgems.Services.OpenAIclient import OpenAIClient


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
        # df_scaled = pd.DataFrame(self.scaler.transform(df), columns=self.features)
        return df

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

    def _build_shap_explainer(self, background: pd.DataFrame = None):
        """
        Build a SHAP explainer appropriate to the model type.

        Ensures that the background DataFrame is purely numeric,
        since SHAP explainer backends generally require numeric input.
        """
        # Defensive conversion: try to coerce to numeric dtype where possible
        if background is not None:
            try:
                background = background.apply(pd.to_numeric, errors="ignore")
                non_numeric_cols = background.select_dtypes(include=["object"]).columns.tolist()
                if len(non_numeric_cols) > 0:
                    print(f"⚠️ Warning: background data still has non-numeric columns: {non_numeric_cols}")
            except Exception as e:
                print(f"⚠️ Could not fully convert background to numeric: {e}")

        try:
            # Prefer TreeExplainer if model is tree-based
            explainer = shap.TreeExplainer(self.model, data=background)
            return explainer
        except Exception as e:
            print(f"TreeExplainer failed: {e}")
            # Try the more general Explainer (e.g. for linear models)
            try:
                explainer = shap.Explainer(self.model, background)
                return explainer
            except Exception as inner_e:
                raise RuntimeError(f"Could not build a SHAP explainer: {inner_e}")

    def compute_shap_for_datapoint(
        self, data_point: Dict, top_n: int = 5, background_samples: pd.DataFrame = None
    ) -> Tuple[dict, List[dict]]:
        df_pre = self.preprocess_input(data_point)

        background = background_samples if background_samples is not None else df_pre

        explainer = self._build_shap_explainer(background=background)

        try:
            shap_result = explainer(df_pre)
        except Exception as e:
            raise RuntimeError("Error computing SHAP values: " + str(e))

        values = getattr(shap_result, "values", None)
        base_values = getattr(shap_result, "base_values", None)

        class_index = None
        try:
            probs = self.model.predict_proba(df_pre)[0]
            class_index = int(np.argmax(probs))
        except Exception:
            class_index = None

        if values is None:
            raise RuntimeError("SHAP returned no values (shap_result.values is None).")

        arr = None
        try:
            if isinstance(values, list) or (
                hasattr(values, "ndim") and getattr(values, "ndim", None) == 3
            ):
                if (
                    class_index is not None
                    and isinstance(values, (list, np.ndarray))
                    and class_index < len(values)
                ):
                    arr = np.array(values[class_index]).reshape(-1)
                else:
                    arr = np.array(values[0]).reshape(-1)
            else:
                # typical shape: (1, n_features)
                arr = np.array(values).reshape(-1)
        except Exception:
            # Last-resort flatten
            arr = np.array(values).reshape(-1)

        base = None
        if base_values is not None:
            try:
                if isinstance(base_values, (list, np.ndarray)):
                    base_arr = np.array(base_values).reshape(-1)
                    if class_index is not None and class_index < base_arr.size:
                        base = float(base_arr[class_index])
                    else:
                        base = float(base_arr[0])
                else:
                    base = float(base_values)
            except Exception:
                base = None

        feature_names = list(df_pre.columns)
        # Defensive trim/pad if mismatched lengths
        min_len = min(len(feature_names), arr.size)
        feature_names = feature_names[:min_len]
        arr = arr[:min_len]

        shap_map = {}
        for fname, sval in zip(feature_names, arr):
            raw_val = data_point.get(fname, None)
            shap_map[fname] = {
                "shap_value": float(sval),
                "abs": float(abs(sval)),
                "raw_value": raw_val,
            }

        sorted_feats = sorted(shap_map.items(), key=lambda x: x[1]["abs"], reverse=True)
        ordered_shap = OrderedDict((k, v) for k, v in sorted_feats)

        top_features = []
        for fname, meta in sorted_feats[:top_n]:
            top_features.append({"feature": fname, **meta})

        shap_summary_dict = {"base_value": base, "features": ordered_shap}
        print(shap_summary_dict)
        return shap_summary_dict, top_features

    async def explain_delay_with_shap_and_llm(
        self, data_point: Dict, top_n: int = 5, background_samples: pd.DataFrame = None
    ) -> Dict:
        """
        Full flow for a single datapoint:
          - Preprocess and predict
          - Compute SHAP explanation
          - Format concise SHAP summary and call the LLM to produce a one-line causal reason if prediction indicates delay

        Returns:
            A dictionary containing:
              - prediction: model predicted class
              - probability: predicted probabilities (list)
              - shap_summary: as returned by compute_shap_for_datapoint
              - top_features: list of top feature dicts
              - llm_message: one-line explanation from the LLM if prediction suggests 'delayed', otherwise None
        """
        df_pre = self.preprocess_input(data_point)
        prediction = self.model.predict(df_pre)[0]
        probability = self.model.predict_proba(df_pre)[0]

        shap_summary, top_features = self.compute_shap_for_datapoint(
            data_point=data_point, top_n=top_n, background_samples=background_samples
        )

        def feature_line(feat_dict):
            sign = "+" if feat_dict["shap_value"] >= 0 else "-"
            return f"{feat_dict['feature']}: {sign}{abs(feat_dict['shap_value']):.4f} (value={feat_dict['raw_value']})"

        shap_text_lines = [feature_line(f) for f in top_features]
        shap_text = "\n".join(shap_text_lines)
        base_val = shap_summary.get("base_value", None)
        base_line = (
            f"base_value: {base_val:.4f}"
            if base_val is not None
            else "base_value: unknown"
        )

        system_prompt = """
        You are an assistant that explains why shipments are delayed using model explanation (SHAP).
        Produce a single-line plain-language reason for why this shipment may be delayed.
        Use the SHAP contributions to cite which features likely pushed the prediction toward delay.
        Keep language simple and concise.
        """

        user_prompt = f"""
        Original datapoint:
        {json.dumps(data_point, default=str)}

        Model prediction: {prediction}
        Class probabilities: {np.array2string(probability, precision=4, separator=', ')}

        SHAP summary (top {top_n} features)
        {base_line}
        {shap_text}

        Produce a ONE-LINE reason (not bullets) for why the shipment may be delayed,
        referencing the most important contributing features and a short causal phrase.
        """

        llm_message = None
        if str(prediction).lower() in ("delayed", "delay", "1", "true", "yes"):
            llm_message = await self.llm.generate_response(system_prompt, user_prompt)

        return {
            "prediction": prediction,
            "probability": (
                probability.tolist() if hasattr(probability, "tolist") else probability
            ),
            "shap_summary": shap_summary,
            "top_features": top_features,
            "llm_message": llm_message,
        }

    async def get_data_for_inference_from_cube(self):
        """
        Fetch dataset for inference from DatasetPreparation, run inference row-by-row,
        compute SHAP + LLM explanation for 'delayed' cases, and collect results.

        Returns:
            List[dict] where each dict contains batch_in_id, title, prediction, message (LLM result), and metadata.
        """
        # 1) Load data using your repository helper (should return a DataFrame)
        get_data = DatasetPreparation(
            customer_name=self.customer_name,
            start_timestamp=self.start_timestamp,
            end_timestamp=self.end_timestamp,
            po_comitted=self.po_comitted,
        ).calculate_target_variable_from_clean_dataset(task="inference")

        # Short system prompt for the LLM is embedded in explain_delay_with_shap_and_llm
        results = []

        # Iterate rows and perform inference + explanation
        for _, row in get_data.iterrows():
            # Convert row to dict and remove target label if present
            data_point = row.to_dict()
            data_point.pop("shipment_classified", None)
            batch_id = data_point.get("batch_in_id")
            batch_number = data_point.get("batch_number", "Unknown")

            # Optional quick sanity check: ensure columns align with model features
            if sorted(list(data_point.keys())) != sorted(list(self.features)):
                # Warn but continue; preprocess_input expects self.features.
                # If mismatch is expected, adapt this check or ensure DatasetPreparation yields correct columns.
                print("Warning: data point keys differ from model features; proceeding anyway.")

            # 1) Basic prediction & feature importance
            prediction, probability, feature_importance = self.predict(data_point)
            print(f"Computed prediction for batch {batch_number}: {prediction}")
            print("Top model feature importances (if available):", feature_importance)

            # 2) If delayed, compute SHAP + call LLM for a one-line causal reason
            if str(prediction).lower() in ("delayed", "delay", "1", "true", "yes"):
                explanation = await self.explain_delay_with_shap_and_llm(data_point, top_n=5)

                results.append(
                    {
                        "batch_in_id": batch_id,
                        "title": "Shipment may be delayed for batch number: " + batch_number,
                        "prediction": explanation["prediction"],
                        "message": explanation["llm_message"],
                        "customer_name": self.customer_name,
                        "vendor_id": data_point.get("vendor_id"),
                        "customer_po": data_point.get("purchase_order"),
                        "sku": data_point.get("sku"),
                        "po_comitted": self.po_comitted,
                        # Optionally include raw SHAP summary for dashboards or auditing
                        # "shap": explanation["shap_summary"],
                        "probability": explanation["probability"],
                    }
                )

        return results
    
    # async def get_data_for_inference_from_cube(self):
    #     """
    #     Fetch data, run inference row-by-row,
    #     and return a list of dictionaries containing
    #     batch_in_id and prediction.
    #     """
    #     # 1️⃣ Load data
    #     get_data = DatasetPreparation(
    #         customer_name=self.customer_name,
    #         start_timestamp=self.start_timestamp,
    #         end_timestamp=self.end_timestamp,
    #         po_comitted=self.po_comitted,
    #     ).calculate_target_variable_from_clean_dataset(task="inference")

    #     system_prompt = """
    #     You are a helpful assistant that will help analyze the shipment issues and get the reason behind the shipment delay in a single line.
    #     Write the reason why the shipment may have been potentially delayed in a single line considering the feature importance that is provided.
    #     Instead of just relaying on the feature importance value alone, try to come up with why those features may have contributed to the delay.
    #     Use Simple language and avoid using too many words.
    #     Here are detail of the features columns that are used for training the model:
    #     - quantity_in: The quantity of the shipment
    #     - seal: The seal of the shipment
    #     - yield_percentage: The yield percentage of the shrimps
    #     - gap_between_due_and_eta: The gap between the due date the bill must be paid and the eta (estimated time of arrival of shipment)
    #     - gap_between_shipped_and_due: The gap between the shipped date(date when shipment is marked as shipped) and the due date(the date when the bill must be paid)
    #     - gap_between_bill_and_shipment: The gap between the bill date(the date when the bill was issued) and the shipped date(the date when shipment is marked as shipped)
    #     - gap_between_eta_shipped: The gap between the eta date(estimated time of arrival of shipment) and the shipped date(the date when shipment is marked as shipped)
    #     - gap_between_eta_bill: The gap between the eta date(estimated time of arrival of shipment) and the bill date(the date when the bill was issued)
    #     - due_gap: The gap between the due date(the date when the bill must be paid) and the bill date(the date when the bill was issued)
    #     - responsiveness: The responsiveness of the shipment (the gap between the bill date and the shipped date)

    #     Flow of the shipment:
    #     first bill is issued along with due date, then shipment is marked as shipped, then eta is estimated, then shipment is received.
        
    #     Example:
    #     Delayed may be due to the following reasons:
    #     - The gap between the due date and the eta is too long
    #     - The gap between the shipped date and the due date is too long
    #     - The gap between the bill date and the shipped date is too long
    #     """
    #     user_prompt = """
    #     Here is the data:
    #     {data_point}
    #     Here is the prediction:
    #     {prediction}
    #     Here is the probability for each class:
    #     {probability}
    #     Here is the feature importance:
    #     {feature_importance}
    #     """

    #     results = []

    #     for _, row in get_data.iterrows():
    #         data_point = row.to_dict()
    #         data_point.pop("shipment_classified", None)
    #         batch_id = data_point.get("batch_in_id")
    #         batch_number = data_point.get("batch_number", "Unknown")
    #         print(sorted(list(data_point.keys())) == sorted(list(self.features)))

    #         # if batch_id is None:
    #         #     continue  # skip if batch_in_id missing

    #         prediction, probability, feature_importance = self.predict(data_point)
    #         print(feature_importance)
    #         print(f"The prediction is: {prediction}")
    #         if prediction == "delayed":
    #             status_message = await self.llm.generate_response(
    #                 system_prompt,
    #                 user_prompt.format(
    #                     data_point=data_point,
    #                     prediction=prediction,
    #                     probability=probability,
    #                     feature_importance=feature_importance,
    #                 ),
    #             )

    #             results.append(
    #                 {
    #                     "batch_in_id": batch_id,
    #                     "title": "Shipment may be delayed for batch number: "
    #                     + batch_number,
    #                     "prediction": prediction,
    #                     "message": status_message,
    #                     "customer_name": self.customer_name,
    #                     "vendor_id": data_point.get("vendor_id"),
    #                     "customer_po": data_point.get("purchase_order"),
    #                     "sku": data_point.get("sku"),
    #                     "po_comitted": self.po_comitted,
    #                 }
    #             )

    #     return results
        for _, row in get_data.iterrows():
            data_point = row.to_dict()
            data_point.pop("shipment_classified", None)
            batch_id = data_point.get("batch_in_id")
            batch_number = data_point.get("batch_number", "Unknown")
            print(sorted(list(data_point.keys())) == sorted(list(self.features)))

            # if batch_id is None:
            #     continue  # skip if batch_in_id missing

            prediction, probability, feature_importance = self.predict(data_point)
            print(feature_importance)
            print(f"The prediction is: {prediction}")
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
        for _, row in get_data.iterrows():
            data_point = row.to_dict()
            data_point.pop("shipment_classified", None)
            batch_id = data_point.get("batch_in_id")
            batch_number = data_point.get("batch_number", "Unknown")
            print(sorted(list(data_point.keys())) == sorted(list(self.features)))

            # if batch_id is None:
            #     continue  # skip if batch_in_id missing

            prediction, probability, feature_importance = self.predict(data_point)
            print(feature_importance)
            print(f"The prediction is: {prediction}")
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
