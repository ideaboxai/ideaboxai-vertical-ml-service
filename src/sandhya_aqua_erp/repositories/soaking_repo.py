from .base_repo import BaseRepository
from ..db_conn import get_sandhya_db_engine
from typing import Optional, List

import pandas as pd


class SoakingRepository(BaseRepository):
    """
    Repository for handling Soaking data preparation.
    """

    db_name = "erpx_dev_production"
    available_tables = [
        "pp_soaking",
        "pp_soaking_lot",
        "pp_soaking_readings",
        "pp_soaking_temperature",
    ]

    def __init__(self):
        self.engine = get_sandhya_db_engine()

    def get_individual_table(
        self, table_name: str = "pp_soaking", column_names: Optional[List[str]] = None
    ):
        if table_name not in self.available_tables:
            raise ValueError(
                f"Table {table_name} is not available in the SoakingRepository."
            )

        query = (
            f"SELECT * FROM {self.db_name}.{table_name} WHERE created_at > '2025-06-01'"
        )

        if column_names:
            columns = ", ".join(column_names)
            query = f"SELECT {columns} FROM {self.db_name}.{table_name} WHERE created_at > '2025-06-01'"

        df = pd.read_sql(query, self.engine)
        return df

    def get_combined_table(self):
        """
        Fetches and returns the combined soaking table with joined data from all related tables.
        """
        query = """
            SELECT
                psl.session_id,
                psl.lot_number,
                psl.grading_lot_id AS session_id_at_grading_stage,
                psl.count AS shrimp_count_per_pound_during_grading_stage,

                ps.tenant_id,
                ps.tub_number,
                ps.sale_order,
                ps.sku,
                ps.additives,
                ps.soak_started_at,
                ps.closed_at AS soaking_closed_at,
                ps.soak_time AS desired_soaking_time_in_minutes,
                ps.status AS soaking_status,
                ps.crate_weight,
                ps.brand,
                ps.process_type,
                ps.soakin_count,

                psr.sequence_number AS soaking_reading_sequence_number,
                psr.weight AS weight_in_kg,
                psr.crates AS crate_count,
                psr.location_id AS soaking_reading_location_id,
                psr.location_name AS soaking_reading_location_name,
                psr.created_at AS soaking_reading_created_date,
                psr.epoch AS soaking_reading_epoch,
                psr.epoch_date AS soaking_reading_epoch_date,

                pst.temperature AS soaking_temp_in_celsius,
                pst.temperature_fault AS soaking_temperature_fault,
                pst.epoch_date AS soaking_temperature_reading_time

            FROM
                erpx_dev_production.pp_soaking_lot psl
            JOIN
                erpx_dev_production.pp_soaking ps
                ON psl.session_id = ps.session_id
            LEFT JOIN
                erpx_dev_production.pp_soaking_readings psr
                ON psr.session_id = ps.session_id AND psr.tub_number = ps.tub_number
            LEFT JOIN
                erpx_dev_production.pp_soaking_temperature pst
                ON pst.session_id = ps.session_id AND pst.tub_number = ps.tub_number
            WHERE
                psl.grading_lot_id > 4144
                AND psr.tub_number IS NOT NULL
                AND psr.tub_number != 0
            ORDER BY
                psl.session_id,
                psr.sequence_number;
        """
        df = pd.read_sql(query, self.engine)
        return df
