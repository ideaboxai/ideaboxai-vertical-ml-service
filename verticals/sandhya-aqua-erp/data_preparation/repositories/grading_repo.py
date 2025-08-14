from .base_repo import BaseRepository
from ..db_conn import get_sandhya_db_engine
from typing import Optional, List

import pandas as pd


class GradingRepository(BaseRepository):
    """
    Repository for handling Grading data preparation.
    """

    db_name = "erpx_dev_production"
    available_tables = ["pp_grading", "pp_grading_grades", "pp_grading_readings"]

    def __init__(self):
        self.engine = get_sandhya_db_engine()

    def get_individual_table(
        self, table_name: str = "pp_grading", column_names: Optional[List[str]] = None
    ):
        if table_name not in self.available_tables:
            raise ValueError(
                f"Table {table_name} is not available in the GradingRepository."
            )

        query = f"SELECT * FROM {self.db_name}.{table_name}"

        if column_names:
            columns = ", ".join(column_names)
            query = f"SELECT {columns} FROM {self.db_name}.{table_name}"

        df = pd.read_sql(query, self.engine)
        return df

    def get_combined_table(self):
        query = """
            SELECT
                pg.session_id,
                pg.lot_id,
                pg.lot_name,
                pg.crate_weight,
                pg.status,
                pg.tenant_id,
                pg.location_id AS grading_location_id,
                pg.created_at AS grading_created_date,
                pg.closed_at AS grading_closed_date,

                ppg.count AS shrimp_count_per_pound,
                ppg.uniformity_ratio,
                ppg.bin_number,
                ppg.created_at AS grading_grade_created_date,

                pgr.sequence_number AS grading_reading_sequence_number,
                pgr.sale_order,
                pgr.product_sku,
                pgr.weight AS weight_in_kg,
                pgr.crates AS crate_count,
                pgr.location_id AS grading_reading_location_id,
                pgr.location_name AS grading_reading_location_name,
                pgr.created_at AS grading_reading_created_date,
                pgr.epoch AS grading_reading_epoch,
                pgr.epoch_date AS grading_reading_epoch_date

            FROM erpx_dev_production.pp_grading pg
                JOIN erpx_dev_production.pp_grading_grades ppg ON ppg.session_id = pg.session_id
                LEFT JOIN erpx_dev_production.pp_grading_readings pgr ON pgr.session_id = ppg.session_id AND pgr.bin_number = ppg.bin_number
            WHERE
                ppg.bin_number IS NOT NULL
                AND ppg.bin_number != 0
                AND pg.session_id > 4144
            ORDER BY
                pg.session_id,
                pgr.sequence_number;
        """
        df = pd.read_sql(query, self.engine)
        return df
