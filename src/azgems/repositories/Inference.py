import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Dict, List, Optional
import shap
import json
import asyncio
from pydantic import BaseModel, Field

from src.azgems.Services.OpenAIclient import OpenAIClient


# Constants matching TrainModel.py
DELAY_THRESHOLD = 5  # for derived classification
USE_LOG_TARGET = True  # log1p transform was used during training


class ShipmentExplanation(BaseModel):
    """Pydantic model for structured LLM explanations."""

    reasonings: List[str] = Field(
        ...,
        description="List of reasonings for the predicted shipment delay. First item should contain how many days of delay was predicted.",
    )


class Inference:
    """
    Inference class for loading trained models and making predictions.

    The model pipeline includes preprocessing, so raw data can be passed directly.
    """

    def __init__(
        self,
        customer_name: str,
        model_path: Optional[str] = None,
        enable_shap: bool = True,
        enable_llm_explanations: bool = True,
    ):
        """
        Initialize the Inference class.

        Args:
            customer_name: Name of the customer (used to construct default model path)
            model_path: Optional custom path to the model. If None, uses default:
                       models/azgems/{customer_name}/ShipmentClassificationModel.joblib
            enable_shap: Whether to enable SHAP explainer (default: True)
            enable_llm_explanations: Whether to enable LLM-based explanations (default: True)
        """
        self.customer_name = customer_name
        self.model = None
        self.model_path = None
        self.explainer = None
        self.enable_shap = enable_shap
        self.enable_llm_explanations = enable_llm_explanations

        # Determine model path
        if model_path is None:
            self.model_path = (
                Path("models")
                / "azgems"
                / customer_name
                / "ShipmentClassificationModel.joblib"
            )
        else:
            self.model_path = Path(model_path)

        # Load model during initialization
        self.load_model()

        # Initialize SHAP explainer if enabled
        if self.enable_shap:
            self._initialize_shap_explainer()

        # Initialize OpenAI client if LLM explanations are enabled
        if self.enable_llm_explanations and OpenAIClient is not None:
            try:
                self.llm_client = OpenAIClient()
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI client: {e}")
                self.enable_llm_explanations = False
                self.llm_client = None
        else:
            self.llm_client = None

    def load_model(self) -> None:
        """
        Load the trained model from disk.

        Raises:
            FileNotFoundError: If the model file doesn't exist
            ValueError: If the model file cannot be loaded
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at: {self.model_path}. "
                f"Please ensure the model has been trained and saved first."
            )

        try:
            self.model = joblib.load(self.model_path)
            print(f"Model loaded successfully from: {self.model_path}")
        except Exception as e:
            raise ValueError(f"Error loading model from {self.model_path}: {str(e)}")

    def _initialize_shap_explainer(self):
        """
        Initialize the SHAP explainer for the loaded model.

        This should be called after the model is loaded.
        """
        if self.model is None:
            raise ValueError("Model must be loaded before initializing SHAP explainer.")

        try:
            # Get the model and preprocessing steps from the pipeline
            if hasattr(self.model, "named_steps"):
                # Pipeline structure: [('preprocess', ...), ('model', ...)]
                model_step = self.model.named_steps.get("model")
                prep_step = self.model.named_steps.get("preprocess")

                if model_step is None or prep_step is None:
                    print(
                        "Warning: Could not find 'model' or 'preprocess' steps in pipeline. SHAP may not work correctly."
                    )
                    return

                # Get feature names from preprocessing step
                try:
                    feature_names = prep_step.get_feature_names_out()
                except AttributeError:
                    # Fallback: try to get from ColumnTransformer or other methods
                    try:
                        feature_names = prep_step.get_feature_names_out()
                    except:
                        print(
                            "Warning: Could not get feature names from preprocessing step."
                        )
                        feature_names = None

                # Create SHAP explainer
                if feature_names is not None:
                    self.explainer = shap.Explainer(
                        model_step, feature_names=feature_names
                    )
                else:
                    self.explainer = shap.Explainer(model_step)

                print("SHAP explainer initialized successfully.")
            else:
                # If model is not a pipeline, create explainer directly
                self.explainer = shap.Explainer(self.model)
                print("SHAP explainer initialized successfully (non-pipeline model).")
        except Exception as e:
            print(f"Warning: Could not initialize SHAP explainer: {e}")
            self.explainer = None

    def predict(self, data: Union[pd.DataFrame, Dict, List[Dict]]) -> np.ndarray:
        """
        Make delay_days predictions on input data.

        Args:
            data: Input data in one of the following formats:
                  - pandas DataFrame
                  - Dictionary (single sample)
                  - List of dictionaries (multiple samples)

        Returns:
            numpy.ndarray: Predicted delay_days (inverse transformed if log was used)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Convert input to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError(
                "Data must be a pandas DataFrame, dictionary, or list of dictionaries."
            )

        # Make predictions using the pipeline (handles preprocessing automatically)
        predictions_log = self.model.predict(df)

        # Apply inverse transform if log1p was used during training
        if USE_LOG_TARGET:
            predictions = np.expm1(predictions_log)
        else:
            predictions = predictions_log

        return predictions

    def predict_delay_days(
        self, data: Union[pd.DataFrame, Dict, List[Dict]]
    ) -> Union[float, np.ndarray]:
        """
        Predict delay_days for input data.

        Args:
            data: Input data in one of the following formats:
                  - pandas DataFrame
                  - Dictionary (single sample)
                  - List of dictionaries (multiple samples)

        Returns:
            float or numpy.ndarray: Predicted delay_days
                                   - float for single sample
                                   - numpy.ndarray for multiple samples
        """
        predictions = self.predict(data)

        # Return scalar for single prediction, array for multiple
        if len(predictions) == 1:
            return float(predictions[0])
        return predictions

    def predict_classification(
        self,
        data: Union[pd.DataFrame, Dict, List[Dict]],
        threshold: Optional[float] = None,
    ) -> Union[int, np.ndarray]:
        """
        Predict classification (delayed or not) based on threshold.

        Args:
            data: Input data in one of the following formats:
                  - pandas DataFrame
                  - Dictionary (single sample)
                  - List of dictionaries (multiple samples)
            threshold: Delay threshold in days (default: DELAY_THRESHOLD)
                      Predictions > threshold are classified as delayed (1)

        Returns:
            int or numpy.ndarray: Binary classification
                                 - 0: Not delayed (delay_days <= threshold)
                                 - 1: Delayed (delay_days > threshold)
                                 - Returns int for single sample, array for multiple
        """
        if threshold is None:
            threshold = DELAY_THRESHOLD

        predictions = self.predict(data)
        classifications = (predictions > threshold).astype(int)

        # Return scalar for single prediction, array for multiple
        if len(classifications) == 1:
            return int(classifications[0])
        return classifications

    def predict_with_details(
        self,
        data: Union[pd.DataFrame, Dict, List[Dict]],
        threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Make predictions and return detailed results including both regression and classification.

        Args:
            data: Input data in one of the following formats:
                  - pandas DataFrame
                  - Dictionary (single sample)
                  - List of dictionaries (multiple samples)
            threshold: Delay threshold in days for classification (default: DELAY_THRESHOLD)

        Returns:
            pandas.DataFrame: Results with columns:
                - predicted_delay_days: Predicted delay in days
                - is_delayed: Binary classification (0/1)
                - delay_category: "Not Delayed" or "Delayed"
        """
        if threshold is None:
            threshold = DELAY_THRESHOLD

        # Convert input to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError(
                "Data must be a pandas DataFrame, dictionary, or list of dictionaries."
            )

        # Get predictions
        delay_predictions = self.predict(df)
        classifications = self.predict_classification(df, threshold=threshold)

        # Create results DataFrame
        results = pd.DataFrame(
            {
                "predicted_delay_days": delay_predictions,
                "is_delayed": classifications,
                "delay_category": pd.Series(classifications)
                .map({0: "Not Delayed", 1: "Delayed"})
                .values,
            }
        )

        return results

    def _transform_data_for_shap(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform input data using the preprocessing step for SHAP computation.

        Args:
            data: Input DataFrame with raw features

        Returns:
            Transformed DataFrame ready for SHAP explainer
        """
        if not hasattr(self.model, "named_steps"):
            return data

        prep_step = self.model.named_steps.get("preprocess")
        if prep_step is None:
            return data

        # Transform the data using the preprocessing step
        transformed_data = prep_step.transform(data)

        # Convert to DataFrame with feature names if available
        try:
            feature_names = prep_step.get_feature_names_out()
            return pd.DataFrame(
                transformed_data, columns=feature_names, index=data.index
            )
        except:
            # If we can't get feature names, return as array (SHAP can handle this)
            return transformed_data

    def get_shap_values(
        self,
        data: Union[pd.DataFrame, Dict, List[Dict]],
        background_data: Optional[pd.DataFrame] = None,
    ) -> Optional[shap.Explanation]:
        """
        Compute SHAP values for the given data.

        Args:
            data: Input data in one of the following formats:
                  - pandas DataFrame
                  - Dictionary (single sample)
                  - List of dictionaries (multiple samples)
            background_data: Optional background dataset for SHAP (if not provided, uses the input data)

        Returns:
            SHAP Explanation object or None if SHAP is disabled or explainer not available
        """
        if not self.enable_shap or self.explainer is None:
            return None

        # Convert input to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError(
                "Data must be a pandas DataFrame, dictionary, or list of dictionaries."
            )

        # Transform data for SHAP
        transformed_data = self._transform_data_for_shap(df)

        # Compute SHAP values
        try:
            shap_values = self.explainer(transformed_data)
            return shap_values
        except Exception as e:
            print(f"Error computing SHAP values: {e}")
            return None

    def _get_llm_system_prompt(self) -> str:
        """Get the system prompt for LLM-based explanations."""
        return """You are an expert data scientist specializing in machine learning model output explanations and the reasoning behind them. 
        Your task is to help explain the predictions of a Gradient Boosting regression model trained to predict shipment delay days.
        When given feature values for a shipment and SHAP values, provide a clear, concise explanation of how features contribute to the predicted delay days.
        Focus on the most impactful features and their influence on the prediction.
        Try to keep the explanation based on the data provided.
        Try to find the root cause behind the delay.
        Avoid using technical jargon; explain in simple terms.
        Try to explain the predictions in bullet points that are **short and concise such that it is quick for customers to grasp**.
        Do not use the feature names directly in the explanation.
        Try to back up the reasonings with the numerical value wherever possible.

        Here is the idea behind the features provided:
        - shipping_duration_days: Expected duration of the shipment in days.
        - lead_time_days: Number of days between order placement and shipment.
        - is_early_delivery: Indicator if the delivery was early (1) or not (0).
        - coo: Country of origin of the shipment.
        - scac: Standard Carrier Alpha Code representing the shipping carrier.
        - tariff_amount: The tariff cost associated with the shipment.
        - ocean_freight: Cost of ocean freight for the shipment.
        - delivery_terms: Terms of delivery (e.g., CY, DDP).
        - po_shipment_terms: Purchase order shipment terms.
        - tariff_type: Type of tariff applied to the shipment.
        - total_bcy: Total cost in base currency.
        - quantity_in: Quantity of items in the shipment.
        - item_sku: Stock Keeping Unit identifier for the item.
        - item_brand: Brand of the item being shipped.
        - item_manufacturer: Manufacturer of the item.
        - item_product_category: Category of the product being shipped.
        - item_size: Size specification of the item.
        - vendor_name: Name of the vendor supplying the item.
        - vendor_avg_delay_days: Average delay days for shipments from this vendor.
        - vendor_shipments: Total number of shipments made by this vendor.
        - vendor_on_time_rate: Percentage of on-time deliveries by this vendor.
        - vendor_p50_delay_days: 50th percentile delay days for this vendor.
        - vendor_p90_delay_days: 90th percentile delay days for this vendor.
        - shipped_date_weekday: Day of the week the shipment was sent (0=Monday, 6=Sunday).
        - shipped_date_month: Month the shipment was sent (1-12).
        - shipped_date_day: Day of the month the shipment was sent (1-31).

        Things to avoid:
        "The day of the week when the shipment was sent may contribute slightly to delays." Instead write "Shipments sent on (name of the day) may experience minor delays due to operational factors based on historical data".

        Expected Outputs Format:
        The output should have the following bullets or points in the output like 
        - Predicted delay
        - Shipping time
        - Vendor on-time rate
        - Vendor Average Past Delay Days
        - Shipped day

        if there is no data available then write "N/A" in the output.
        
        Example 1:
        Predicted delay: ~30 days
        Shipping time: 139 days (high uncertainty)
        Vendor on-time rate: 5% (frequent delays)
        Lead time: 233 days (error-prone)
        Ship day: Friday (weekend hold risk)

        Example 2:
        Predicted delay: ~5 days
        Ship day: Monday → operational backlog
        Vendor on-time rate: 33% (high risk)
        Shipping time: 73 days → unexpected issues
        High tariffs/freight: Customs & logistics delays
        """

    async def generate_llm_explanation(
        self,
        data: Union[pd.DataFrame, Dict, List[Dict]],
        predicted_delay: float,
        shap_values: Optional[shap.Explanation] = None,
    ) -> Optional[List[str]]:
        """
        Generate LLM-based explanation for a prediction using SHAP values.

        Args:
            data: Input data used for prediction
            predicted_delay: The predicted delay in days
            shap_values: Optional SHAP values (will be computed if not provided)

        Returns:
            List of explanation strings or None if LLM is disabled
        """
        if not self.enable_llm_explanations or self.llm_client is None:
            return None

        # Convert input to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError(
                "Data must be a pandas DataFrame, dictionary, or list of dictionaries."
            )

        # Get SHAP values if not provided
        if shap_values is None and self.enable_shap:
            shap_values = self.get_shap_values(df)

        # Build user prompt
        user_prompt = f"""Given the following feature values for a shipment:
            {df.to_dict(orient='records')[0]}

            The model predicted a delay of {predicted_delay:.2f} days.
            Please explain how the key features influenced this prediction."""

        if shap_values is not None:
            # Add SHAP values to the prompt
            try:
                # Get SHAP values for the first sample
                if hasattr(shap_values, "values") and len(shap_values.values) > 0:
                    shap_vals = shap_values.values[0]
                    # Get feature names if available
                    if hasattr(shap_values, "feature_names"):
                        feature_names = shap_values.feature_names
                    else:
                        feature_names = [f"feature_{i}" for i in range(len(shap_vals))]

                    # Create a dictionary of feature names to SHAP values
                    shap_dict = dict(zip(feature_names, shap_vals))
                    user_prompt += f"\n\nSHAP values for the features are:\n{json.dumps(shap_dict, indent=2)}"
            except Exception as e:
                print(f"Warning: Could not include SHAP values in prompt: {e}")

        try:
            # Generate explanation using structured output
            system_prompt = self._get_llm_system_prompt()
            explanation = await self.llm_client.generate_formatted_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format=ShipmentExplanation,
                temperature=0.7,
            )

            if explanation and hasattr(explanation, "reasonings"):
                return explanation.reasonings
            return None
        except Exception as e:
            print(f"Error generating LLM explanation: {e}")
            return None

    def predict_with_explanations(
        self,
        data: Union[pd.DataFrame, Dict, List[Dict]],
        include_shap: bool = True,
        include_llm: bool = True,
        threshold: Optional[float] = None,
    ) -> Dict:
        """
        Make predictions with SHAP values and LLM-based explanations.

        Args:
            data: Input data in one of the following formats:
                  - pandas DataFrame
                  - Dictionary (single sample)
                  - List of dictionaries (multiple samples)
            include_shap: Whether to include SHAP values (default: True)
            include_llm: Whether to include LLM explanations (default: True)
            threshold: Delay threshold in days for classification (default: DELAY_THRESHOLD)

        Returns:
            Dictionary containing:
                - predictions: Predicted delay days
                - classifications: Binary classification (if threshold provided)
                - shap_values: SHAP Explanation object (if include_shap=True)
                - llm_explanation: List of explanation strings (if include_llm=True)
        """
        if threshold is None:
            threshold = DELAY_THRESHOLD

        # Convert input to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError(
                "Data must be a pandas DataFrame, dictionary, or list of dictionaries."
            )

        # Get predictions
        predictions = self.predict(df)

        # Initialize result dictionary
        result = {
            "predictions": predictions,
            "classifications": (
                (predictions > threshold).astype(int) if threshold else None
            ),
        }

        # Get SHAP values if requested
        if include_shap and self.enable_shap:
            shap_values = self.get_shap_values(df)
            result["shap_values"] = shap_values
        else:
            shap_values = None
            result["shap_values"] = None

        # Get LLM explanations if requested (only if there is a delay)
        if include_llm and self.enable_llm_explanations:
            classifications_arr = result["classifications"]
            
            # Helper function to check if a prediction indicates a delay
            def is_delayed_prediction(idx: int) -> bool:
                """Check if prediction at index idx indicates a delay."""
                if classifications_arr is not None:
                    if isinstance(classifications_arr, np.ndarray):
                        return bool(classifications_arr[idx])
                    else:
                        # Single prediction case - classifications_arr is a scalar
                        return bool(classifications_arr) if idx == 0 else False
                else:
                    # If no classifications, check if prediction > threshold
                    delay_threshold = threshold if threshold is not None else 0.0
                    return predictions[idx] > delay_threshold
            
            # For single prediction, generate explanation only if delayed
            if len(predictions) == 1:
                if is_delayed_prediction(0):
                    llm_explanation = asyncio.run(
                        self.generate_llm_explanation(
                            df, float(predictions[0]), shap_values
                        )
                    )
                    result["llm_explanation"] = llm_explanation
                else:
                    result["llm_explanation"] = None
            else:
                # For multiple predictions, generate explanations only for delayed ones
                explanations = []
                for idx in range(len(predictions)):
                    if is_delayed_prediction(idx):
                        single_df = df.iloc[[idx]]
                        single_shap = None
                        if shap_values is not None and hasattr(shap_values, "values"):
                            # Extract SHAP values for this sample
                            single_shap = shap.Explanation(
                                values=shap_values.values[idx : idx + 1],
                                base_values=(
                                    shap_values.base_values[idx : idx + 1]
                                    if hasattr(shap_values, "base_values")
                                    else None
                                ),
                                data=(
                                    shap_values.data[idx : idx + 1]
                                    if hasattr(shap_values, "data")
                                    else None
                                ),
                                feature_names=(
                                    shap_values.feature_names
                                    if hasattr(shap_values, "feature_names")
                                    else None
                                ),
                            )
                        explanation = asyncio.run(
                            self.generate_llm_explanation(
                                single_df, float(predictions[idx]), single_shap
                            )
                        )
                        explanations.append(explanation)
                    else:
                        explanations.append(None)
                result["llm_explanation"] = explanations
        else:
            result["llm_explanation"] = None

        return result


if __name__ == "__main__":
    import pandas as pd
    from time import time

    # ---- Config ----
    DELAY_THRESHOLD = 5.0  # generate LLM explanation only if pred_delay_days > 5

    # 0) Load data
    from src.azgems.repositories.DatasetPreparation import DatasetPreparation
    
    dataset_preparation = DatasetPreparation()
    df = dataset_preparation.clean_dataset(method="inference").dropna()

    if df.empty:
        print("No rows to score. Exiting.")
        raise SystemExit(0)

    print(df.info())

    # 1) Init model/inferencer
    inferencer = Inference(customer_name="Walmart")
    want_details = True
    want_shap_default = inferencer.enable_shap
    want_llm_default  = inferencer.enable_llm_explanations

    results = []
    n_rows = len(df)
    t0 = time()
    print(f"Scoring {n_rows} rows (row-by-row)…")

    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        row_out = {"row_index": idx}

        try:
            # --- Regression ---
            y_delay = inferencer.predict_delay_days(row_dict)
            if hasattr(y_delay, "__len__") and not isinstance(y_delay, (str, bytes)):
                y_delay = y_delay[0]
            y_delay = float(y_delay)
            row_out["pred_delay_days"] = y_delay

            # --- Classification ---
            y_cls = inferencer.predict_classification(row_dict)
            if hasattr(y_cls, "__len__") and not isinstance(y_cls, (str, bytes)):
                y_cls = y_cls[0]
            row_out["is_delayed"] = int(y_cls)

            # --- Details (optional) ---
            if want_details:
                try:
                    details_df = inferencer.predict_with_details(pd.DataFrame([row]))
                    if isinstance(details_df, pd.DataFrame) and not details_df.empty:
                        for k, v in details_df.iloc[0].items():
                            row_out[f"detail__{k}"] = v
                except Exception as e_det:
                    row_out["detail__error"] = str(e_det)

            # --- LLM/SHAP only if predicted delay > threshold ---
            if (y_delay is not None) and (y_delay > DELAY_THRESHOLD) and (want_llm_default or want_shap_default):
                try:
                    exp = inferencer.predict_with_explanations(
                        pd.DataFrame([row]),
                        include_shap=want_shap_default,     # set True to also compute SHAP when delayed
                        include_llm=want_llm_default        # LLM only when delayed
                    )

                    # SHAP flag (don’t store huge arrays inline)
                    if want_shap_default and exp.get("shap_values") is not None:
                        row_out["shap__available"] = 1
                    else:
                        row_out["shap__available"] = 0

                    # LLM explanation (single-row)
                    if want_llm_default and exp.get("llm_explanation") is not None:
                        llm = exp["llm_explanation"]
                        row_out["llm_explanation"] = (llm[0] if isinstance(llm, list) and llm else llm)
                except Exception as e_exp:
                    row_out["explain__error"] = str(e_exp)
            else:
                row_out["llm_skipped_reason"] = (
                    "prediction_not_over_threshold" if y_delay <= DELAY_THRESHOLD else "llm_disabled"
                )
                row_out["shap__available"] = 0

            row_out["status"] = "ok"

        except Exception as e:
            row_out["status"] = "error"
            row_out["error"] = str(e)

        results.append(row_out)

        # progress
        if (len(results) % 50) == 0:
            elapsed = time() - t0
            print(f"  processed {len(results)}/{n_rows} rows in {elapsed:.1f}s…")

    elapsed = time() - t0
    print(f"Done. Processed {n_rows} rows in {elapsed:.1f}s.")

    # 2) Assemble results
    results_df = pd.DataFrame(results).set_index("row_index").sort_index()
    final_df = results_df  # or df.join(results_df, how="left") to include inputs

    print("\nPreview (first 10):")
    cols_show = [c for c in ["pred_delay_days", "is_delayed", "llm_explanation", "status", "error"] if c in final_df.columns]
    print(final_df[cols_show].head(10).to_string())

    out_path = "predictions_rowwise.csv"
    final_df.to_csv(out_path)
    print(f"\nSaved row-wise predictions to {out_path}")

