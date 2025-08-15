from .base_repo import BaseRepository
from ...db_conn import get_sandhya_db_engine
from typing import Optional, List

import pandas as pd


class CookingRepository(BaseRepository):
    """
    Repository for handling Cooking data preparation.
    """

    db_name = "erpx_dev_production"
    available_tables = ["pp_cooking", "pp_cooking_lot", "pp_cooking_readings"]

    def __init__(self):
        self.engine = get_sandhya_db_engine()

    def get_individual_table(
        self, table_name: str = "pp_cooking", column_names: Optional[List[str]] = None
    ):
        if table_name not in self.available_tables:
            raise ValueError(
                f"Table {table_name} is not available in the CookingRepository."
            )

        query = f"SELECT * FROM {self.db_name}.{table_name}"

        if column_names:
            columns = ", ".join(column_names)
            query = f"SELECT {columns} FROM {self.db_name}.{table_name}"

        df = pd.read_sql(query, self.engine)
        return df

    def get_combined_table(self):
        pass
