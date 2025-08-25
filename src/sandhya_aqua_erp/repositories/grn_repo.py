from .base_repo import BaseRepository
from ..db_conn import get_sandhya_db_engine
from typing import Optional, List

import pandas as pd


class GRNRepository(BaseRepository):
    """
    Repository for handling Goods Receipt Note (GRN) data preparation.
    """

    db_name = "erpx_dev_rm_procurement"

    def __init__(self):
        self.engine = get_sandhya_db_engine()

    def get_individual_table(
        self, table_name: str = "grn_items", column_names: Optional[List[str]] = None
    ):
        query = (
            f"SELECT * FROM {self.db_name}.{table_name} WHERE created_at > '2025-06-01'"
        )

        if column_names:
            columns = ", ".join(column_names)
            query = f"SELECT {columns} FROM {self.db_name}.{table_name} WHERE created_at > '2025-06-01'"

        df = pd.read_sql(query, self.engine)
        return df

    def get_farmer_related_data(
        self, interval: Optional[str] = None, exact_date_time: Optional[str] = None
    ):
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
                ii.farmer,
                s.name AS farmer_name,

                AVG(TIMESTAMPDIFF(HOUR, i.indent_date, g.grn_date)) AS avg_time_to_grn,

                SUM(gi.quantity) AS total_actual_quantity,                                         -- SUMMING ALL ACTUAL QUANITTY FROM FARMER RATHER TO AVERAGING

                AVG(ABS(gi.quantity - ii.expected_qty)) AS avg_quantity_variance,

                GROUP_CONCAT(DISTINCT i.indent_id) AS indent_ids,
                COUNT(DISTINCT i.indent_id) AS indent_count,

                GROUP_CONCAT(DISTINCT gi.plant_lot_number) AS plant_lot_numbers,
                COUNT(DISTINCT gi.plant_lot_number) AS lots_supplied,

                AVG(hl.grading_yield) AS avg_grading_yield

            FROM
                erpx_dev_rm_procurement.indent_items AS ii
            JOIN
                erpx_dev_rm_procurement.indent AS i ON ii.indent_id = i.indent_id
            LEFT JOIN
                erpx_dev_rm_procurement.grn AS g ON g.indent_no = i.indent_id
            LEFT JOIN
                erpx_dev_rm_procurement.grn_items AS gi ON gi.grn_id = g.grn_id
            LEFT JOIN
                erpx_dev.suppliers AS s ON s.supplier_id = ii.farmer
            LEFT JOIN
                hl_cte AS hl ON hl.grn_id = gi.grn_id
            """

        if interval:
            query += f" WHERE g.grn_date >= NOW() - INTERVAL {interval} "
        elif exact_date_time:
            query += f" WHERE g.grn_date >= '{exact_date_time}' "

        query += """
            GROUP BY
                ii.farmer
            ORDER BY
                ii.farmer;
        """

        df = pd.read_sql(query, self.engine)
        return df

    def get_combined_table(self):
        pass
