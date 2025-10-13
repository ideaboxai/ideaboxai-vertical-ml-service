import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression as MLmodel
import joblib
import os
from sklearn.model_selection import cross_val_score

from src.azgems.repositories.get_dataset import DatasetPreparation


class TrainModel:
    def __init__(self, customer_name):
        self.dataset_prep = DatasetPreparation()
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.customer_name = customer_name

    def configure_model(self):
        """Initialize the Logistic Regression model."""
        self.model = MLmodel(class_weight="balanced", random_state=42)

    def encode_dataset(self, df: pd.DataFrame, target_col: str = "shipment_classified"):
        """Encode categorical columns and scale numerical ones."""
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        print("while training")
        print(X.info())
        # Label encode categorical columns
        for col in X.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le

        # Standardize numerical columns
        self.scaler = StandardScaler()
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)

        return X_scaled, y

    def train_and_save_model(self, save_path: str = None):
        """Train the logistic regression model and save it."""
        df = self.dataset_prep.calculate_target_variable_from_clean_dataset()
        target_col = "shipment_classified"

        # Encode dataset
        X, y = self.encode_dataset(df, target_col)

        # Configure model
        self.configure_model()

        # --- Cross-validation ---
        cv_scores = cross_val_score(
            self.model, X, y, cv=5, scoring="accuracy"
        )  # 5-fold CV
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.3f}")

        # --- Train on full dataset ---
        self.model.fit(X, y)

        if save_path is None:
            save_path = f"models/{self.customer_name}/logistic_regression.pkl"

        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)

        # Save model, scaler, and encoders
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "encoders": self.label_encoders,
                "features": list(X.columns),
            },
            save_path,
        )

        print(f"Model trained on full dataset and saved to {save_path}")


if __name__ == "__main__":
    trainer = TrainModel(customer_name="Walmart")
    trainer.train_and_save_model()
