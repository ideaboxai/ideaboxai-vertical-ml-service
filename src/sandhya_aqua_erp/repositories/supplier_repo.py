from .base_repo import BaseRepository
from ..db_conn import get_sandhya_db_engine
from typing import Optional

import pandas as pd


class SupplierRepository(BaseRepository):
    """
    Repository for handling Goods Receipt Note (GRN) data preparation.
    """

    db_name = "erpx_dev_rm_procurement"

    def __init__(self):
        self.engine = get_sandhya_db_engine()

    def get_farmer_related_data(
        self, interval: Optional[str] = None, exact_date_time: Optional[str] = None
    ):
        query = """
                WITH farmer_service AS (
                    SELECT
                        ii.farmer,
                        MIN(g.created_at) AS first_grn_date,
                        TIMESTAMPDIFF(MONTH, MIN(g.created_at), CURDATE()) AS farmer_service_period_in_months
                    FROM erpx_dev_rm_procurement.indent_items AS ii
                    JOIN erpx_dev_rm_procurement.indent AS i ON ii.indent_id = i.indent_id
                    LEFT JOIN erpx_dev_rm_procurement.grn AS g ON g.indent_no = i.indent_id
                    WHERE g.created_at IS NOT NULL
                    GROUP BY ii.farmer
                ),
                hl_cte AS (
                    SELECT
                        g.grn_id,
                        g.grn_item_id,
                        g.created_at GRN_Created_dt,
                        p.lot_name,
                        g.count as hon_count,
                        g.quantity as hon_weight,
                        p.created_at as Grading_Created_dt,
                        FLOOR(SUM(v.weight) - (p.crate_weight * COUNT(v.crates))) AS hl_weight,
                        ((SUM(v.weight) - (p.crate_weight * COUNT(v.crates))) / g.quantity) * 100 AS grading_yield,
                        GROUP_CONCAT(DISTINCT v.sale_order ORDER BY v.sale_order ASC SEPARATOR ', ') AS grading_sale_orders
                    FROM
                        erpx_dev_rm_procurement.grn_items g
                    LEFT JOIN erpx_dev_production.pp_grading p
                        ON g.plant_lot_number = p.lot_name
                    LEFT JOIN erpx_dev_production.pp_grading_readings v
                        ON p.session_id = v.session_id
                    -- WHERE g.created_at >= DATE_SUB(CURDATE(), INTERVAL 2 MONTH)
                    GROUP BY
                        g.grn_item_id,
                        g.created_at,
                        p.lot_name,
                        g.count,
                        g.quantity,
                        p.created_at
                    ),
                grading_count_cte AS (
                    SELECT
                        lot_name,
                        SUM(net_grading_weight * grading_count * 2.20462) / SUM(net_grading_weight) AS weighted_avg_shrimp_count_per_kg
                    FROM (
                        SELECT
                            pg.lot_name,
                            pgr.count AS grading_count,
                            SUM(pgr.weight) - (pg.crate_weight * SUM(pgr.crates)) AS net_grading_weight
                        FROM pp_grading pg
                        LEFT JOIN pp_grading_readings pgr
                            ON pg.session_id = pgr.session_id
                        GROUP BY pg.session_id, pg.lot_name
                    ) t
                    GROUP BY lot_name
                )
                SELECT
                    ii.farmer,
                    s.name AS farmer_name,
                    fl.first_grn_date,
                    fl.farmer_service_period_in_months,

                    AVG(TIMESTAMPDIFF(HOUR, i.indent_date, g.grn_date)) AS avg_time_to_grn,

                    SUM(gi.quantity) AS total_actual_quantity,

                    AVG(ABS(gi.quantity - ii.expected_qty)) AS avg_quantity_variance,
                    AVG(ABS(gi.count - ii.expected_count)) AS abs_avg_count_variance_indent_grn,

                    AVG(gc.weighted_avg_shrimp_count_per_kg) AS avg_shrimp_count_at_grading,
                    AVG(gc.weighted_avg_shrimp_count_per_kg - gi.count) AS avg_count_variance_grn_grading,

                    -- GROUP_CONCAT(DISTINCT i.indent_id) AS indent_ids,
                    COUNT(DISTINCT i.indent_id) AS indent_count,

                    -- GROUP_CONCAT(DISTINCT gi.plant_lot_number) AS plant_lot_numbers

                    AVG(hl.grading_yield) AS avg_grading_yield

                FROM
                    erpx_dev_rm_procurement.indent_items AS ii
                LEFT JOIN
                    erpx_dev_rm_procurement.grn_items AS gi ON gi.indent_item_id = ii.indent_item_id
                LEFT JOIN
                    erpx_dev_rm_procurement.indent AS i ON ii.indent_id = i.indent_id
                LEFT JOIN
                    erpx_dev_rm_procurement.grn AS g ON g.grn_id  = gi.grn_id
                LEFT JOIN
                    erpx_dev.suppliers AS s ON s.supplier_id = ii.farmer
                LEFT JOIN
                    hl_cte AS hl ON hl.grn_item_id = gi.grn_item_id
                LEFT JOIN
                    farmer_service AS fl ON fl.farmer = ii.farmer
                LEFT JOIN
                    grading_count_cte AS gc ON gc.lot_name = gi.plant_lot_number
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

    def insert_farmer_rankings(self):
        pass

    def get_combined_table(self):
        return super().get_combined_table()

    def get_individual_table(self):
        return super().get_individual_table()
