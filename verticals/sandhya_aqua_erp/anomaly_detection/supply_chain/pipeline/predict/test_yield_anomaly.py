from predict_cross_stage import predict
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

query = """
            WITH hl_cte AS (
                SELECT
                    gi.grn_id,
                    gi.created_at AS grn_created_at,
                    pg.lot_name,
                    gi.count AS hon_count,
                    gi.quantity AS hon_weight,
                    pg.created_at AS grading_created_at,
                    FLOOR(SUM(pgr.weight) - (pg.crate_weight * COUNT(pgr.crates))) AS hl_weight,
                    ( (SUM(pgr.weight) - (pg.crate_weight * COUNT(pgr.crates))) / NULLIF(gi.quantity, 0) ) * 100 AS grading_yield,
                    GROUP_CONCAT(DISTINCT pgr.sale_order ORDER BY pgr.sale_order ASC SEPARATOR ', ') AS grading_sale_orders
                FROM erpx_dev_rm_procurement.grn_items gi
                LEFT JOIN erpx_dev_production.pp_grading pg
                    ON gi.plant_lot_number = pg.lot_name
                LEFT JOIN erpx_dev_production.pp_grading_readings pgr
                    ON pg.session_id = pgr.session_id
                GROUP BY
                    gi.grn_id,
                    gi.created_at,
                    pg.lot_name,
                    gi.count,
                    gi.quantity,
                    pg.created_at
            )
            SELECT
                gi.plant_lot_number,
                ii.expected_count AS indent_count,
                ii.expected_qty AS indent_quantity,
                0 AS indent_yield,
                gi.quantity AS grn_quantity,
                gi.count AS grn_count,
                (gi.quantity / NULLIF(ii.expected_qty, 0)) * 100 AS grn_yield,
                hc.grading_yield,
                hc.grading_sale_orders,
                hc.hl_weight,
                hc.hon_count,
                hc.hon_weight,
                i.indent_id,
                g.grn_id,
                i.created_at AS indent_created_at
            FROM erpx_dev_rm_procurement.indent i
            LEFT JOIN erpx_dev_rm_procurement.indent_items ii
                ON i.indent_id = ii.indent_id
            LEFT JOIN erpx_dev_rm_procurement.grn_items gi
                ON ii.indent_item_id = gi.indent_item_id
                AND i.indent_id = gi.indent_id
            LEFT JOIN erpx_dev_rm_procurement.grn g
                ON gi.grn_id = g.grn_id
            LEFT JOIN hl_cte hc
                ON gi.grn_id = hc.grn_id
                AND hc.lot_name = gi.plant_lot_number;
        """

df = pd.read_sql(query, sqlalchemy.create_engine(sandhya_erp_db_url))

pd.set_option("display.max_rows", None)
data = predict(cross_stage_name="grn_grading_yield_data", data=df)
print(data[0:2].T)
