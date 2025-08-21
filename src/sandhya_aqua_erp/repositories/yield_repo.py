import pandas as pd
from ..db_conn import get_sandhya_db_engine
from typing import Optional


class YieldRepository:
    """
    Repository for handling Yield data preparation.
    """

    def __init__(self):
        self.engine = get_sandhya_db_engine()

    def get_grn_grading_yield_data(
        self, interval: Optional[str] = None, exact_date_time: Optional[str] = None
    ) -> pd.DataFrame:
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
                AND hc.lot_name = gi.plant_lot_number
        """
        if interval and not exact_date_time:
            query += f" WHERE hc.grading_created_at >= DATE_SUB(CURDATE(), INTERVAL {interval})"
        elif exact_date_time and not interval:
            query += f" WHERE hc.grading_created_at > '{exact_date_time}'"
        elif interval and exact_date_time:
            raise ValueError(
                "Specify only one of 'interval' or 'exact_date_time', not both."
            )

        df = pd.read_sql(query, self.engine)
        return df

    def get_soaking_yield_data(
        self, interval: Optional[str] = None, exact_date_time: Optional[str] = None
    ) -> pd.DataFrame:
        query = """
                WITH hl_cte AS (
                    SELECT
                        g.grn_id,
                        lot_name,
                        p.created_at AS grading_created_at,
                        SUM(v.weight) - (p.crate_weight * COUNT(v.crates)) AS grading_net_weight,
                        g.quantity AS grn_quantity,
                        ((SUM(v.weight) - (p.crate_weight * COUNT(v.crates))) / g.quantity) * 100 AS grading_yield
                    FROM
                        erpx_dev_rm_procurement.grn_items g
                    LEFT JOIN erpx_dev_production.pp_grading p
                        ON g.plant_lot_number = p.lot_name
                    LEFT JOIN erpx_dev_production.pp_grading_readings v
                        ON p.session_id = v.session_id
                    WHERE
                        p.lot_name IS NOT NULL
                    GROUP BY
                        g.grn_id,
                        lot_name,
                        g.quantity,
                        p.crate_weight
                ),
                grading_cte AS (
                    SELECT
                        p.lot_name,
                        p.session_id,
                        pgr.sale_order,
                        pgr.count AS grading_count,
                        CEIL(SUM(pgr.weight)) AS grading_ct_wt,
                        MAX(p.crate_weight) * SUM(pgr.crates) AS crate_weight,
                        CEIL(SUM(pgr.weight)) - (MAX(p.crate_weight) * SUM(pgr.crates)) AS grading_weight
                    FROM
                        erpx_dev_production.pp_grading_readings pgr
                    JOIN erpx_dev_production.pp_grading p
                        ON pgr.session_id = p.session_id
                    WHERE
                        p.created_at >= DATE_SUB(CURDATE(), INTERVAL 2 MONTH)
                        AND pgr.sale_order IS NOT NULL
                    GROUP BY
                        p.lot_name,
                        p.session_id,
                        pgr.sale_order,
                        pgr.count
                ),
                soaking_cte AS (
                    SELECT
                        sl.lot_number,
                        sl.count AS soaking_cnt,
                        ps.unit_id,
                        ps.sku,
                        GROUP_CONCAT(
                            DISTINCT ps.sale_order
                            ORDER BY ps.sale_order ASC
                            SEPARATOR ', '
                        ) AS soaking_sale_orders,
                        CEIL(SUM(p.weight) - (SUM(p.crates) * MAX(ps.crate_weight))) AS soaking_weight
                    FROM
                        erpx_dev_production.pp_soaking_readings p
                    JOIN erpx_dev_production.pp_soaking_lot sl
                        ON p.session_id = sl.session_id
                    JOIN erpx_dev_production.pp_soaking ps
                        ON p.session_id = ps.session_id
                    WHERE
                        ps.created_at >= DATE_SUB(CURDATE(), INTERVAL 2 MONTH)
                    GROUP BY
                        sl.lot_number,
                        sl.count
                ),
                cooking_cte AS (
                    SELECT
                        pcl.lot_number,
                        pcl.count AS cooking_count,
                        pc.sale_order,
                        pc.sku,
                        pc.cooking_temp,
                        pc.chilling_temp
                    FROM
                        erpx_dev_production.pp_cooking_readings pcr
                    JOIN erpx_dev_production.pp_cooking_lot pcl
                        ON pcr.session_id = pcl.session_id
                    JOIN erpx_dev_production.pp_cooking pc
                        ON pcr.session_id = pc.session_id
                    WHERE
                        pc.created_at >= DATE_SUB(CURDATE(), INTERVAL 2 MONTH)
                    GROUP BY
                        pcl.lot_number,
                        pcl.count,
                        pc.sale_order,
                        pc.sku,
                        pc.cooking_temp,
                        pc.chilling_temp
                )
                SELECT
                    gd.lot_name,
                    gd.sale_order  AS grading_sale_order,
                    sd.soaking_sale_orders,
                    hc.grading_yield,
                    hc.grading_net_weight,
                    hc.grn_quantity,
                    gd.grading_count,
                    gd.grading_weight,
                    sd.soaking_cnt,
                    sd.soaking_weight,
                    (sd.soaking_weight / NULLIF(gd.grading_weight, 0)) * 100 AS soaking_yield,
                    cd.cooking_temp,
                    cd.chilling_temp
                FROM
                    grading_cte gd
                LEFT JOIN hl_cte hc
                    ON gd.lot_name = hc.lot_name
                LEFT JOIN soaking_cte sd
                    ON gd.lot_name = sd.lot_number
                    AND gd.grading_count = sd.soaking_cnt
                LEFT JOIN cooking_cte cd
                    ON gd.lot_name = cd.lot_number
                    AND gd.grading_count = cd.cooking_count
            """
        if interval and not exact_date_time:
            query += f" WHERE hc.grading_created_at >= DATE_SUB(CURDATE(), INTERVAL {interval})"
        elif exact_date_time and not interval:
            query += f" WHERE hc.grading_created_at > '{exact_date_time}'"
        elif interval and exact_date_time:
            raise ValueError(
                "Specify only one of 'interval' or 'exact_date_time', not both."
            )

        df = pd.read_sql(query, self.engine)
        return df

    def get_packing_yield_data(
        self, interval: Optional[str] = None, exact_date_time: Optional[str] = None
    ) -> pd.DataFrame:
        query = """
            SELECT
                soak.unit_id,

                soak.sale_order soak_sale_order,
                pack.customer_po pack_sale_order,
                soak.brand,
                soak.sku,
                soak.weight soak_weight,
                pack.pak_wt,
                Round(100 * pack.pak_wt / soak.weight, 1) pack_yield
            FROM
                (
                SELECT
                    ps.unit_id,
                    ps.sku,
                    Group_concat(DISTINCT ps.sale_order) sale_order,
                    Group_concat(DISTINCT ps.brand) brand,
                    Round(Sum(sr.weight - ( ps.crate_weight * sr.crates )), 0) weight
                FROM
                    erpx_dev_production.pp_soaking_lot sl
                LEFT JOIN erpx_dev_production.pp_soaking_readings sr
                                ON
                    sl.session_id = sr.session_id
                LEFT JOIN erpx_dev_production.pp_soaking ps
                                ON
                    sl.session_id = ps.session_id
                GROUP BY
                    ps.unit_id,
                    ps.sku
                ORDER BY
                    1,
                    2,
                    3,
                    4
                ) soak
            LEFT JOIN (
                SELECT
                    pa.unit_id,
                    pa.product_sku,
                    pa.created_at AS packing_created_at,
                    Group_concat(DISTINCT Ifnull(
                                    NULLIF(Ltrim(pa.customer_po), '')
                                                        , 'DUMMY')
                                                    )
                                                    customer_po,
                    -- pa.packing_style,
                    Sum(Substr(pa.packing_style, 1, 2) * net_weight)
                                            wt,
                    -- case when pa.unit_id = 'NlSqOgl7yGzFB2eLAADPl' then 'unit2' else 'unit1' end plant,
                    -- pa.no_of_bags,
                    Sum(pouch.pouch_qty)
                                                    pouch_qty,
                    Sum(ma.mc_qty)
                                                    mc_qty,
                    Sum(Round(ma.mc_qty * Substr(pa.packing_style, 1, 2) *
                                            net_weight))
                                                    pak_wt
                FROM
                    erpx_dev_production.pp_packing pa
                INNER JOIN (
                                                /* MC Qty - Master Carton Quantity number of packets processed for specific unit, session line etc.. */
                    SELECT
                        session_id,
                        unit_id,
                        Count(sequence_number) AS mc_qty
                    FROM
                        erpx_dev_production.pp_packing_master_readings
                    GROUP BY
                        session_id,
                        unit_id) ma
                                            ON
                    pa.session_id = ma.session_id
                    AND pa.unit_id = ma.unit_id
                INNER JOIN (
                                                /* Pouch Qty - Pouch Quantity number of pouches processed for specific SKU, session, line etc.. */
                    SELECT
                        session_id,
                        unit_id,
                        Count(id) pouch_qty
                    FROM
                        erpx_dev_production.pp_packing_reading
                    GROUP BY
                        session_id,
                        unit_id) pouch
                                            ON
                    ma.session_id = pouch.session_id
                    AND ma.unit_id = pouch.unit_id
                GROUP BY
                    1,
                    2
                ORDER BY
                    pa.customer_po,
                    pa.unit_id) pack
                        ON
                soak.unit_id = pack.unit_id
                -- and soak.sale_order = case when trim(pack.customer_po) ='' then 'DUMMY'  else pack.customer_po end
                AND pack.product_sku = soak.sku
        """
        if interval and not exact_date_time:
            query += f" WHERE pack.packing_created_at >= DATE_SUB(CURDATE(), INTERVAL {interval})"
        elif exact_date_time and not interval:
            query += f" WHERE pack.packing_created_at > '{exact_date_time}'"
        elif interval and exact_date_time:
            raise ValueError(
                "Specify only one of 'interval' or 'exact_date_time', not both."
            )

        query += " ORDER BY 1, 5, 2, 3"
        df = pd.read_sql(query, self.engine)
        return df
