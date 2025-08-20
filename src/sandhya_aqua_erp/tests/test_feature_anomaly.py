"""
To run this test: Use the command:
python -m src.sandhya_aqua_erp.tests.test_feature_anomaly
"""

from src.sandhya_aqua_erp.anomaly_detection.supply_chain.pipeline.predict.predict_feature_wise import (
    predict,
)
import pandas as pd
import os
from dotenv import load_dotenv
import sqlalchemy

load_dotenv()

DB_HOST = os.getenv("SANDHYA_ERP_DB_HOST")
DB_PORT = os.getenv("SANDHYA_ERP_DB_PORT")
DB_USERNAME = os.getenv("SANDHYA_ERP_DB_USERNAME")
DB_PASSWORD = os.getenv("SANDHYA_ERP_DB_PASSWORD")
DB_NAME = os.getenv("SANDHYA_ERP_DB_NAME")

sandhya_erp_db_url = (
    f"mysql+pymysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# query = """
# SELECT * FROM erpx_dev_rm_procurement.grn_items
# ORDER BY grn_item_id DESC
# LIMIT 1
# """

query = """
SELECT * FROM erpx_dev_production.pp_grading_readings
ORDER BY session_id ASC
LIMIT 100
"""

df = pd.read_sql(query, sqlalchemy.create_engine(sandhya_erp_db_url))

# data = predict(
#     database_name="erpx_dev_rm_procurement",
#     table_name="grn_items",
#     feature_name="count",
#     data=df
# )
# print(data)

data = predict(
    database_name="erpx_dev_production",
    table_name="pp_grading_readings",
    feature_name="weight",
    data=df,
)
print(data)
