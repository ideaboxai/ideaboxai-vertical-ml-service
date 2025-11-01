import os
import sys
import warnings
from pathlib import Path
from contextlib import contextmanager
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    classification_report,
    confusion_matrix,
)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder


from src.azgems.repositories.DatasetPreparation import DatasetPreparation


TARGET = "delay_days"
DELAY_THRESHOLD = 5  # for derived classification check
USE_LOG_TARGET = True  # log1p transform to tame skew   -> log1p = log(1+x)
USE_SAMPLE_WEIGHTS = True  # emphasize larger delays during training
RANDOM_STATE = 42

# Best parameters found from GridSearchCV
BEST_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.01,
    "max_depth": 5,
    "min_samples_leaf": 2,
    "subsample": 0.8,
}


@contextmanager
def suppress_resource_tracker_warnings():
    """
    Context manager to suppress Python 3.13 multiprocessing resource tracker
    ValueError warnings that come from subprocess stderr.
    """
    import sys
    
    # Save original stderr
    original_stderr = sys.stderr
    
    try:
        # Create a filter class that suppresses resource tracker ValueError messages
        class FilteredStderr:
            def __init__(self, original_stderr):
                self.original_stderr = original_stderr
            
            def write(self, message):
                # Check if message contains resource tracker warnings
                if message and isinstance(message, str):
                    if ("resource_tracker" in message.lower() and 
                        "unknown resource type semlock" in message.lower()):
                        return  # Suppress these messages
                # Also check for ValueError patterns
                if message and ("ValueError" in str(message) and 
                                "semlock" in str(message).lower()):
                    return  # Suppress these messages
                # Write to original stderr
                self.original_stderr.write(message)
                return len(message) if message else 0
            
            def flush(self):
                self.original_stderr.flush()
            
            def __getattr__(self, name):
                # Delegate all other attributes to original stderr
                return getattr(self.original_stderr, name)
        
        sys.stderr = FilteredStderr(original_stderr)
        yield
    finally:
        # Restore original stderr
        sys.stderr = original_stderr


class TrainModel:
    def __init__(self, customer_name, model_params=None):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.customer_name = customer_name
        self.dataset_prep = DatasetPreparation(customer_name=customer_name)
        # Use provided model_params or default to BEST_PARAMS
        self.model_params = model_params if model_params is not None else BEST_PARAMS.copy()
        # Initialize instance variables for data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_train_raw = None
        self.y_test_raw = None
        self.w_train = None
        self.inv = None
        self.pipeline = None
        self.best_model = None
        self.model_save_path = None

    def configure_model(self):
        """Initialize the Gradient Boosting Regressor model with best parameters."""
        self.model = GradientBoostingRegressor(
            n_estimators=self.model_params["n_estimators"],
            learning_rate=self.model_params["learning_rate"],
            max_depth=self.model_params["max_depth"],
            min_samples_leaf=self.model_params["min_samples_leaf"],
            subsample=self.model_params["subsample"],
            random_state=RANDOM_STATE,
        )

    def encode_dataset(self, df: pd.DataFrame, target_col: str = "delay_days"):
        """Encode categorical columns and scale numerical ones."""
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train_raw, y_test_raw = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )

        # Store raw targets for evaluation
        self.y_train_raw = y_train_raw
        self.y_test_raw = y_test_raw

        # target transform (log1p helps a lot with right-skew)
        if USE_LOG_TARGET:
            y_train = np.log1p(y_train_raw)
            y_test = np.log1p(y_test_raw)
            self.inv = np.expm1
        else:
            y_train = y_train_raw.copy()
            y_test = y_test_raw.copy()
            self.inv = lambda z: z

        # Store transformed targets
        self.y_train = y_train
        self.y_test = y_test

        # regression "balancing": sample weights that up-weight long delays
        if USE_SAMPLE_WEIGHTS:
            rng = (y_train_raw.max() - y_train_raw.min()) or 1.0
            self.w_train = 1.0 + (y_train_raw - y_train_raw.min()) / rng  # ~[1,2]
            # for stronger emphasis use: np.exp((y_train_raw - y_train_raw.min()) / rng)
        else:
            self.w_train = None

        # Store features
        self.X_train = X_train
        self.X_test = X_test

        num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

        column_transformer = ColumnTransformer(
            transformers=[
                ("num", "passthrough", num_cols),
                (
                    "cat",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                    cat_cols,
                ),
            ],
            remainder="drop",
        )

        return column_transformer

    def setup_pipeline(self, column_transformer):
        """Setup the pipeline for the model."""
        if self.model is None:
            raise ValueError(
                "Model must be configured before setting up pipeline. Call configure_model() first."
            )
        self.pipeline = Pipeline(
            steps=[
                ("preprocess", column_transformer),
                ("model", self.model),
            ]
        )

    def train_model(self):
        """Train the model with the configured best parameters."""
        if self.pipeline is None:
            raise ValueError(
                "Pipeline must be set up before training. Call setup_pipeline() first."
            )
        if self.X_train is None or self.y_train is None:
            raise ValueError(
                "Data must be encoded before training. Call encode_dataset() first."
            )

        # fit (pass sample_weight if enabled)
        fit_kwargs = (
            {"model__sample_weight": self.w_train} if self.w_train is not None else {}
        )

        print(f"Training model with parameters: {self.model_params}")
        self.pipeline.fit(self.X_train, self.y_train, **fit_kwargs)

        self.best_model = self.pipeline
        print("Model training complete.")
        return self.best_model

    def save_model(self, model_path: str = None):
        """
        Save the trained model to disk.

        Args:
            model_path: Optional custom path to save the model. If None, uses default path:
                       models/azgems/{customer_name}/ShipmentClassificationModel.joblib

        Returns:
            str: Path where the model was saved
        """
        if self.best_model is None:
            raise ValueError("No trained model found. Train the model first before saving.")

        # Determine model save path
        if model_path is None:
            # Default path: models/azgems/{customer_name}/ShipmentClassificationModel.joblib
            model_dir = Path("models") / "azgems" / self.customer_name
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "ShipmentClassificationModel.joblib"
        else:
            model_path = Path(model_path)
            model_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the model using joblib (recommended for scikit-learn models)
        joblib.dump(self.best_model, model_path)
        self.model_save_path = str(model_path)
        print(f"Model saved successfully to: {self.model_save_path}")
        return str(model_path)

    def evaluate_model_for_regression(self, best_gbr=None):
        """Evaluate the model."""
        if best_gbr is None:
            if self.best_model is None:
                raise ValueError(
                    "No model provided and no best_model stored. Provide a model or train first."
                )
            best_gbr = self.best_model

        if self.X_test is None or self.y_test_raw is None:
            raise ValueError("Test data not available. Call encode_dataset() first.")
        if self.inv is None:
            raise ValueError(
                "Inverse transform function not available. Call encode_dataset() first."
            )

        y_pred_test = self.inv(best_gbr.predict(self.X_test))
        y_true_test = self.y_test_raw
        mae = mean_absolute_error(y_true_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_true_test, y_pred_test))
        r2 = r2_score(y_true_test, y_pred_test)
        print(f"MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")
        return mae, rmse, r2, y_pred_test

    def evaluate_model_for_classification(self, best_gbr=None):
        """Evaluate the model."""
        if best_gbr is None:
            if self.best_model is None:
                raise ValueError(
                    "No model provided and no best_model stored. Provide a model or train first."
                )
            best_gbr = self.best_model

        if self.X_test is None or self.y_test_raw is None:
            raise ValueError("Test data not available. Call encode_dataset() first.")
        if self.inv is None:
            raise ValueError(
                "Inverse transform function not available. Call encode_dataset() first."
            )

        # Get predictions first
        y_pred_test = self.inv(best_gbr.predict(self.X_test))

        pred_is_delayed = (y_pred_test > DELAY_THRESHOLD).astype(int)
        actual_is_delayed = (self.y_test_raw > DELAY_THRESHOLD).astype(int)
        print("\nDerived classification (threshold > 5 days):")
        print(confusion_matrix(actual_is_delayed, pred_is_delayed))
        print(classification_report(actual_is_delayed, pred_is_delayed, digits=3))
        return confusion_matrix(
            actual_is_delayed, pred_is_delayed
        ), classification_report(actual_is_delayed, pred_is_delayed, digits=3)

    def run_full_pipeline(
        self, df: pd.DataFrame = None, target_col: str = "delay_days"
    ):
        """
        Execute the complete training and evaluation pipeline.

        Args:
            df: DataFrame with the training data. If None, will use dataset_prep.clean_dataset()
            target_col: Name of the target column (default: "delay_days")

        Returns:
            dict: Dictionary containing:
                - best_model: The trained best model
                - model_save_path: Path where the model was saved (models/azgems/{customer_name}/ShipmentClassificationModel.joblib)
                - regression_metrics: Tuple of (mae, rmse, r2, y_pred_test)
                - classification_metrics: Tuple of (confusion_matrix, classification_report)
        """
        print("=" * 60)
        print("Starting Full Training Pipeline")
        print("=" * 60)

        # Step 1: Load data if not provided
        if df is None:
            print("\n[1/7] Loading dataset...")
            df = self.dataset_prep.clean_dataset()
            print(f"Loaded dataset with shape: {df.shape}")
            print(df.info())
        else:
            print(f"\n[1/7] Using provided dataset with shape: {df.shape}")

        # Step 2: Configure model
        print("\n[2/7] Configuring model...")
        self.configure_model()
        print("Model configured successfully.")

        # Step 3: Encode dataset
        print("\n[3/7] Encoding dataset and splitting train/test...")
        column_transformer = self.encode_dataset(df=df, target_col=target_col)
        print(
            f"Dataset encoded. Train: {len(self.X_train)}, Test: {len(self.X_test)}"
        )

        # Step 4: Setup pipeline
        print("\n[4/7] Setting up preprocessing pipeline...")
        self.setup_pipeline(column_transformer)
        print("Pipeline setup complete.")

        # Step 5: Train model
        print("\n[5/7] Training model...")
        best_model = self.train_model()

        # Step 6: Save model
        print("\n[6/7] Saving model...")
        model_path = self.save_model()

        # Step 7: Evaluate model (regression and classification)
        print("\n[7/7] Evaluating model performance...")
        print("\n--- Regression Metrics ---")
        regression_metrics = self.evaluate_model_for_regression()

        print("\n--- Classification Metrics ---")
        classification_metrics = self.evaluate_model_for_classification()

        print("\n" + "=" * 60)
        print("Pipeline Complete!")
        print("=" * 60)

        return {
            "best_model": best_model,
            "model_save_path": model_path,
            "regression_metrics": regression_metrics,
            "classification_metrics": classification_metrics,
        }


if __name__ == "__main__":
    train_model = TrainModel(customer_name="Walmart")
    results = train_model.run_full_pipeline()
