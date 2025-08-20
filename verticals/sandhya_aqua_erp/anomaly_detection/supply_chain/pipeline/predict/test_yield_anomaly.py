from predict_cross_stage import predict
import pandas as pd
import os
from dotenv import load_dotenv
import sys
from datetime import datetime

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../"))
)

from verticals.sandhya_aqua_erp.data_preparation.repositories.yield_repo import (
    YieldRepository,
)

load_dotenv()

DB_HOST = os.getenv("SANDHYA_ERP_DB_HOST")
DB_PORT = os.getenv("SANDHYA_ERP_DB_PORT")
DB_USERNAME = os.getenv("SANDHYA_ERP_DB_USERNAME")
DB_PASSWORD = os.getenv("SANDHYA_ERP_DB_PASSWORD")
DB_NAME = os.getenv("SANDHYA_ERP_DB_NAME")

sandhya_erp_db_url = (
    f"mysql+pymysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

yield_repo = YieldRepository()

yield_data_mapper = {
    "grn_grading_yield_data": yield_repo.get_grn_grading_yield_data(),
    "soaking_yield_data": yield_repo.get_soaking_yield_data(),
    "packing_yield_data": yield_repo.get_packing_yield_data(),
}

for cross_stage_name, queried_data in yield_data_mapper.items():
    df = queried_data
    print(f"Processing {cross_stage_name} with {len(df)} records")

    # Batch prediction for the entire DataFrame
    predictions = predict(cross_stage_name=cross_stage_name, data=df)

    # If predict returns a DataFrame or Series, merge with df
    if isinstance(predictions, pd.DataFrame):
        result_df = pd.concat(
            [df.reset_index(drop=True), predictions.reset_index(drop=True)], axis=1
        )
    else:
        df["prediction"] = predictions
        result_df = df

    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "data/sandhya-aqua-erp/output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{cross_stage_name}_anomaly_results_{timestamp}.csv"
    result_df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")
