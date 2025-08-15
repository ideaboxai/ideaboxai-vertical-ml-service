from .base_repo import BaseRepository
from ...db_conn import get_sandhya_db_engine
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
        query = f"SELECT * FROM {self.db_name}.{table_name}"

        if column_names:
            columns = ", ".join(column_names)
            query = f"SELECT {columns} FROM {self.db_name}.{table_name}"

        df = pd.read_sql(query, self.engine)
        return df

    def get_combined_table(self):
        pass
