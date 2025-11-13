import json
import pandas as pd
from ..db_conn import get_sandhya_db_engine
from typing import Optional
from src.shared.clients.cubejs_client import CubeJSClient


class YieldRepository:
    """
    Repository for handling Yield data preparation.
    """

    def __init__(self):
        self.engine = get_sandhya_db_engine()

    # def get_grn_grading_yield_data(
    #     self,
    #     timestamp1: str,
    #     timestamp2: Optional[str] = None,
    #     interval: Optional[str] = None,
    #     operator: str = None,
    # ):
    #     base_query = """
    #         WITH hl_cte AS (
    #             SELECT
    #                 gi.grn_id,
    #                 gi.created_at AS grn_created_at,
    #                 pg.lot_name,
    #                 gi.count AS hon_count,
    #                 gi.quantity AS hon_weight,
    #                 pg.created_at AS grading_created_at,
    #                 FLOOR(SUM(pgr.weight) - (pg.crate_weight * COUNT(pgr.crates))) AS hl_weight,
    #                 ( (SUM(pgr.weight) - (pg.crate_weight * COUNT(pgr.crates))) / NULLIF(gi.quantity, 0) ) * 100 AS grading_yield,
    #                 GROUP_CONCAT(DISTINCT pgr.sale_order ORDER BY pgr.sale_order ASC SEPARATOR ', ') AS grading_sale_orders
    #             FROM erpx_dev_rm_procurement.grn_items gi
    #             LEFT JOIN erpx_dev_production.pp_grading pg
    #                 ON gi.plant_lot_number = pg.lot_name
    #             LEFT JOIN erpx_dev_production.pp_grading_readings pgr
    #                 ON pg.session_id = pgr.session_id
    #             GROUP BY
    #                 gi.grn_id,
    #                 gi.created_at,
    #                 pg.lot_name,
    #                 gi.count,
    #                 gi.quantity,
    #                 pg.created_at
    #         )
    #         SELECT
    #             gi.plant_lot_number,
    #             ii.expected_count AS indent_count,
    #             ii.expected_qty AS indent_quantity,
    #             0 AS indent_yield,
    #             gi.quantity AS grn_quantity,
    #             gi.count AS grn_count,
    #             (gi.quantity / NULLIF(ii.expected_qty, 0)) * 100 AS grn_yield,
    #             hc.grading_yield,
    #             hc.grading_sale_orders,
    #             hc.hl_weight,
    #             hc.hon_count,
    #             hc.hon_weight,
    #             i.indent_id,
    #             g.grn_id,
    #             i.created_at AS indent_created_at
    #         FROM erpx_dev_rm_procurement.indent i
    #         LEFT JOIN erpx_dev_rm_procurement.indent_items ii
    #             ON i.indent_id = ii.indent_id
    #         LEFT JOIN erpx_dev_rm_procurement.grn_items gi
    #             ON ii.indent_item_id = gi.indent_item_id
    #             AND i.indent_id = gi.indent_id
    #         LEFT JOIN erpx_dev_rm_procurement.grn g
    #             ON gi.grn_id = g.grn_id
    #         LEFT JOIN hl_cte hc
    #             ON gi.grn_id = hc.grn_id
    #             AND hc.lot_name = gi.plant_lot_number
    #     """

    #     query = self._build_date_filter(
    #         base_query=base_query,
    #         timestamp_column="hc.grading_created_at",
    #         timestamp1=timestamp1,
    #         timestamp2=timestamp2,
    #         interval=interval,
    #         operator=operator
    #     )

    #     df = pd.read_sql(query, self.engine)
    #     return df

    # def get_soaking_yield_data(
    #     self,
    #     timestamp1: str,
    #     timestamp2: Optional[str] = None,
    #     interval: Optional[str] = None,
    #     operator: str = None,
    # ) -> pd.DataFrame:
    #     base_query = """
    #         WITH grading_cte AS (
    #             SELECT
    #                 p.lot_name,
    #                 p.session_id,
    #                 pgr.sale_order,
    #                 pgr.count AS grading_count,
    #                 CEIL(SUM(pgr.weight)) AS grading_ct_wt,
    #                 MAX(p.crate_weight) * SUM(pgr.crates) AS crate_weight,
    #                 CEIL(SUM(pgr.weight)) - (MAX(p.crate_weight) * SUM(pgr.crates)) AS grading_weight
    #             FROM
    #                 erpx_dev_production.pp_grading_readings pgr
    #             JOIN erpx_dev_production.pp_grading p
    #                 ON pgr.session_id = p.session_id
    #             WHERE
    #                 pgr.sale_order IS NOT NULL
    #             GROUP BY
    #                 p.lot_name,
    #                 p.session_id,
    #                 pgr.sale_order,
    #                 pgr.count
    #         ),
    #         soaking_cte AS (
    #             SELECT
    #                 sl.lot_number,
    #                 sl.count AS soaking_cnt,
    #                 ps.unit_id,
    #                 ps.sku,
    #                 ps.sale_order AS soaking_sale_order,
    #                 CEIL(SUM(p.weight) - (SUM(p.crates) * MAX(ps.crate_weight))) AS soaking_weight,
    #                 ps.created_at AS soaking_created_at
    #             FROM
    #                 erpx_dev_production.pp_soaking_readings p
    #             JOIN erpx_dev_production.pp_soaking_lot sl
    #                 ON p.session_id = sl.session_id
    #             JOIN erpx_dev_production.pp_soaking ps
    #                 ON p.session_id = ps.session_id
    #             GROUP BY
    #                 sl.lot_number,
    #                 sl.count,
    #                 ps.sale_order
    #         ),
    #         cooking_cte AS (
    #             SELECT
    #                 pcl.lot_number,
    #                 pcl.count AS cooking_count,
    #                 pc.sale_order,
    #                 pc.sku,
    #                 pc.cooking_temp,
    #                 pc.chilling_temp
    #             FROM
    #                 erpx_dev_production.pp_cooking_readings pcr
    #             JOIN erpx_dev_production.pp_cooking_lot pcl
    #                 ON pcr.session_id = pcl.session_id
    #             JOIN erpx_dev_production.pp_cooking pc
    #                 ON pcr.session_id = pc.session_id
    #             GROUP BY
    #                 pcl.lot_number,
    #                 pcl.count,
    #                 pc.sale_order,
    #                 pc.sku,
    #                 pc.cooking_temp,
    #                 pc.chilling_temp
    #         )
    #         SELECT
    #             gd.lot_name,
    #             gd.sale_order AS grading_sale_order,
    #             gd.grading_count,
    #             gd.grading_weight,
    #             sd.soaking_cnt,
    #             sd.soaking_weight,
    #             (sd.soaking_weight / NULLIF(gd.grading_weight, 0)) * 100 AS soaking_yield,
    #             cd.cooking_temp,
    #             cd.chilling_temp
    #         FROM
    #             grading_cte gd
    #         LEFT JOIN soaking_cte sd
    #             ON gd.lot_name = sd.lot_number
    #             AND gd.grading_count = sd.soaking_cnt
    #         LEFT JOIN cooking_cte cd
    #             ON gd.lot_name = cd.lot_number
    #             AND gd.grading_count = cd.cooking_count
    #     """

    #     query = self._build_date_filter(
    #         base_query=base_query,
    #         timestamp_column="sd.soaking_created_at",
    #         timestamp1=timestamp1,
    #         timestamp2=timestamp2,
    #         interval=interval,
    #         operator=operator
    #     )

    #     df = pd.read_sql(query, self.engine)
    #     return df

    # def get_packing_yield_data(
    #     self,
    #     timestamp1: str,
    #     timestamp2: Optional[str] = None,
    #     interval: Optional[str] = None,
    #     operator: str = None,
    # ) -> pd.DataFrame:
    #     base_query = """
    #         select soak.unit_id,soak.sale_order soak_sale_order,pack.customer_po pack_sale_order,
    #             soak.brand, soak.sku,
    #             soak.weight soak_weight, pack.pak_wt, pack.pouch_qty, pack.mc_qty ,
    #             round(100 * pack.pak_wt / soak.weight, 1) pack_yield
    #         from
    #             (
    #             select
    #             ps.unit_id, ps.sku,
    #             group_concat(distinct ps.sale_order) sale_order,
    #             group_concat(distinct ps.brand) brand,
    #             round(sum(sr.weight - (ps.crate_weight * sr.crates)), 0) weight
    #         from
    #             erpx_dev_production.pp_soaking_lot sl
    #             left join erpx_dev_production.pp_soaking_readings sr
    #         on
    #             sl.session_id = sr.session_id
    #             left join erpx_dev_production.pp_soaking ps
    #         on
    #             sl.session_id = ps.session_id
    #         where
    #             closed_AT between '2025-07-30 00:00:00.100' and '2025-07-30 23:59:59.000'
    #         group by
    #             ps.unit_id, ps.sku
    #         order by 1, 2, 3, 4
    #         ) soak
    #         left join
    #         (
    #           SELECT
    #             pa.unit_id, pa.product_sku,
    #             pa.created_at AS packing_created_at,
    #             group_concat(distinct ifnull(nullif(ltrim(pa.customer_po), ''), 'DUMMY')) customer_po,
    #             sum(substr(pa.packing_style, 1, 2) * net_weight) wt,
    #             sum(pouch.pouch_qty) pouch_qty,
    #             sum(ma.mc_qty) mc_qty ,
    #             sum(round(ma.mc_qty * substr(pa.packing_style, 1, 2) * net_weight)) pak_wt
    #         FROM
    #             erpx_dev_production.pp_packing pa
    #         inner join (
    #             select session_id, unit_id , count(sequence_number) as mc_qty
    #             from erpx_dev_production.pp_packing_master_readings
    #             where (created_at between '2025-07-30 06:00:00.000' and '2025-07-31 06:00:00.000')
    #             group by session_id, unit_id) ma
    #         on pa.session_id = ma.session_id and pa.unit_id = ma.unit_id
    #         inner join (
    #             select session_id, unit_id, count(id) pouch_qty
    #             from erpx_dev_production.pp_packing_reading
    #             where (created_at between '2025-07-30 06:00:00.000' and '2025-07-31 06:00:00.000')
    #             group by session_id, unit_id) pouch
    #         on ma.session_id = pouch.session_id and ma.unit_id = pouch.unit_id
    #         group by 1, 2
    #         order by pa.customer_po, pa.unit_id
    #         ) pack
    #         on
    #             soak.unit_id = pack.unit_id
    #             and pack.product_sku = soak.sku
    #     """

    #     query = self._build_date_filter(
    #         base_query=base_query,
    #         timestamp_column="pack.packing_created_at",
    #         timestamp1=timestamp1,
    #         timestamp2=timestamp2,
    #         interval=interval,
    #         operator=operator
    #     )

    #     query += " ORDER BY 1, 5, 2, 3"
    #     df = pd.read_sql(query, self.engine)
    #     return df

    # def get_cooking_yield_data(
    #     self,
    #     timestamp1: str = None,
    #     timestamp2: Optional[str] = None,
    #     interval: Optional[str] = None,
    #     operator: str = None,
    # ):
    #     base_query = """
    #     SELECT
    #         `c_o_o_k_i_n_g`.lot_number `c_o_o_k_i_n_g__plant_lot_number`,
    #         `c_o_o_k_i_n_g`.cooking_count `c_o_o_k_i_n_g__count_per_pound`,
    #         `c_o_o_k_i_n_g`.sku `c_o_o_k_i_n_g__sku`,
    #         `c_o_o_k_i_n_g`.sale_order `c_o_o_k_i_n_g__sale_order`,
    #         `c_o_o_k_i_n_g`.time_minutes `c_o_o_k_i_n_g__time_minutes`,
    #         `c_o_o_k_i_n_g`.cooking_temp `c_o_o_k_i_n_g__temperature`,
    #         `c_o_o_k_i_n_g`.unit_id `c_o_o_k_i_n_g__unit_id`,
    #         avg(`c_o_o_k_i_n_g`.cooking_temp) `c_o_o_k_i_n_g__cooking_temp`,
    #         avg(`c_o_o_k_i_n_g`.chilling_temp) `c_o_o_k_i_n_g__chilling_temp`,
    #         avg(`c_o_o_k_i_n_g`.water_zone_temp) `c_o_o_k_i_n_g__water_zone_temp`
    #         FROM
    #         (
    #             SELECT
    #             pcl.lot_number,
    #             pcl.count AS cooking_count,
    #             pc.sale_order,
    #             pc.sku,
    #             pc.unit_id,
    #             pc.created_at,
    #             pc.updated_at,
    #             TIMESTAMPDIFF(MINUTE, pc.started_at, pc.closed_at) AS time_minutes,
    #             MAX(
    #                 CASE
    #                 WHEN pcr.temperature_type = '1' THEN pcr.temperature
    #                 END
    #             ) AS cooking_temp,
    #             MAX(
    #                 CASE
    #                 WHEN pcr.temperature_type = '3' THEN pcr.temperature
    #                 END
    #             ) AS chilling_temp,
    #             MAX(
    #                 CASE
    #                 WHEN pcr.temperature_type = '2'
    #                 OR pcr.temperature_type = '4' THEN pcr.temperature
    #                 END
    #             ) AS water_zone_temp
    #             FROM
    #             erpx_dev_production.pp_cooking_readings pcr
    #             JOIN erpx_dev_production.pp_cooking_lot pcl ON pcr.session_id = pcl.session_id
    #             JOIN erpx_dev_production.pp_cooking pc ON pcr.session_id = pc.session_id
    #             WHERE
    #             pc.created_at >= DATE_SUB(CURDATE(), INTERVAL 2 MONTH)
    #             GROUP BY
    #             pcl.lot_number,
    #             pcl.count,
    #             pc.sale_order,
    #             pc.sku
    #         ) AS `c_o_o_k_i_n_g`
    #         GROUP BY 1, 2, 3, 4, 5, 6, 7
    #     """

    #     query = self._build_date_filter(
    #         base_query=base_query,
    #         timestamp_column="`c_o_o_k_i_n_g`.created_at",
    #         timestamp1=timestamp1,
    #         timestamp2=timestamp2,
    #         interval=interval,
    #         operator=operator
    #     )

    #     df = pd.read_sql(query, self.engine)
    #     return df

    # @staticmethod
    # def _build_date_filter(
    #     base_query: str,
    #     timestamp_column: str,
    #     timestamp1: Optional[str] = None,
    #     timestamp2: Optional[str] = None,
    #     interval: Optional[str] = None,
    #     operator: Optional[str] = None,
    # ) -> str:
    #     """
    #     Build date filter clauses for SQL queries.

    #     Args:
    #         base_query: The base SQL query to append filters to
    #         timestamp_column: The column name to filter on (e.g., 'hc.grading_created_at')
    #         timestamp1: Primary timestamp for filtering
    #         timestamp2: Secondary timestamp for range operations
    #         interval: MySQL interval string (e.g., '1 DAY', '2 MONTH')
    #         operator: Date comparison operator

    #     Returns:
    #         str: Query with appropriate WHERE/AND clauses added

    #     Raises:
    #         ValueError: For invalid parameters or operator combinations
    #     """
    #     query = base_query.strip()
    #     has_where = "WHERE" in query.upper()

    #     if interval and timestamp1:
    #         raise ValueError("Specify only one of 'interval' or 'timestamp1', not both.")

    #     if interval and not timestamp1:
    #         parts = interval.strip().split()
    #         if len(parts) != 2:
    #             raise ValueError("Invalid interval format. Use format like '1 DAY' or '2 MONTH'")

    #         number, unit = parts
    #         try:
    #             int(number)
    #         except ValueError:
    #             raise ValueError("Invalid interval number")

    #         # Validate unit is a known MySQL interval unit
    #         valid_units = ['MICROSECOND', 'SECOND', 'MINUTE', 'HOUR', 'DAY', 'WEEK', 'MONTH', 'QUARTER', 'YEAR']
    #         if unit.upper() not in valid_units:
    #             raise ValueError(f"Invalid interval unit: {unit}")

    #         connector = " WHERE " if not has_where else " AND "
    #         query += f"{connector}{timestamp_column} >= DATE_SUB(CURDATE(), INTERVAL {number} {unit})"
    #         has_where = True

    #     elif timestamp1 and not interval:
    #         connector = " WHERE " if not has_where else " AND "
    #         query += f"{connector}{timestamp_column} > '{timestamp1}'"
    #         has_where = True

    #     if timestamp2:
    #         connector = " AND " if has_where else " WHERE "
    #         query += f"{connector}{timestamp_column} < '{timestamp2}'"
    #         has_where = True

    #     if operator and timestamp1:
    #         connector = " AND " if has_where else " WHERE "

    #         if operator == "equals":
    #             query += f"{connector}DATE({timestamp_column}) = DATE('{timestamp1}')"
    #         elif operator == "notEquals":
    #             query += f"{connector}DATE({timestamp_column}) != DATE('{timestamp1}')"
    #         elif operator == "beforeDate":
    #             query += f"{connector}DATE({timestamp_column}) < DATE('{timestamp1}')"
    #         elif operator == "beforeOrOnDate":
    #             query += f"{connector}DATE({timestamp_column}) <= DATE('{timestamp1}')"
    #         elif operator == "afterDate":
    #             query += f"{connector}DATE({timestamp_column}) > DATE('{timestamp1}')"
    #         elif operator == "afterOrOnDate":
    #             query += f"{connector}DATE({timestamp_column}) >= DATE('{timestamp1}')"
    #         elif operator == "inDateRange" and timestamp2:
    #             query += f"{connector}DATE({timestamp_column}) BETWEEN DATE('{timestamp1}') AND DATE('{timestamp2}')"
    #         elif operator == "notInDateRange" and timestamp2:
    #             query += f"{connector}DATE({timestamp_column}) NOT BETWEEN DATE('{timestamp1}') AND DATE('{timestamp2}')"
    #         elif operator in ["inDateRange", "notInDateRange"] and not timestamp2:
    #             raise ValueError(f"Operator '{operator}' requires timestamp2 parameter")
    #         else:
    #             raise ValueError(f"Invalid operator: {operator}")
    #     elif operator and not timestamp1:
    #         raise ValueError("Operator specified but no timestamp1 provided")

    #     return query

    def get_grn_grading_yield_data(
        self,
        timestamp1: str = None,
        timestamp2: Optional[str] = None,
        interval: Optional[str] = None,
        operator: str = None,
    ):
        cube_client = CubeJSClient()
        query_params = {
            "dimensions": [
                "INDENT_GRN.plant_lot_number",
                "INDENT_GRN.species",
                "INDENT_GRN.count_per_pound",
                "INDENT_GRN.yield",
                "INDENT_GRN.updated_at",
                "INDENT_GRN.hl_weight",
            ],
            "order": {"INDENT_GRN.created_at": "desc"},
            "filters": [],
        }
        params = self._build_date_filter(
            query_param=query_params,
            timestamp_column="INDENT_GRN.updated_at",
            timestamp1=timestamp1,
            timestamp2=timestamp2,
            operator=operator,
        )
        cube_url = cube_client.get_base_url()
        params = {"query": json.dumps(params)}
        response = cube_client.api_call(
            url=cube_url, query_params=params, api_request_method="GET"
        )
        data = response.json()
        if not data or data.get("data") is None:
            return pd.DataFrame()
        df = pd.DataFrame(data["data"])
        return df

    def get_soaking_yield_data(
        self,
        timestamp1: str = None,
        timestamp2: Optional[str] = None,
        interval: Optional[str] = None,
        operator: str = None,
    ):
        cube_client = CubeJSClient()
        query_params = {
            "dimensions": [
                "SOAKING_ALERT.plant_lot_number",
                "SOAKING_ALERT.count_per_pound",
                "GRADING.sku",
                "SOAKING_ALERT.soak_time",
                "SOAKING_ALERT.unit_name",
                "SOAKING_ALERT.updated_at",
                "SOAKING_ALERT.soaking_weight",
            ],
            "order": {"SOAKING_ALERT.created_at": "desc"},
            "measures": ["GRADING.soaking_yield"],
            "filters": [{"member": "SOAKING_ALERT.sku", "operator": "set"}],
        }

        params = self._build_date_filter(
            query_param=query_params,
            timestamp_column="SOAKING_ALERT.updated_at",
            timestamp1=timestamp1,
            timestamp2=timestamp2,
            operator=operator,
        )
        cube_url = cube_client.get_base_url()
        params = {"query": json.dumps(params)}
        response = cube_client.api_call(
            url=cube_url, query_params=params, api_request_method="GET"
        )
        data = response.json()
        if not data or data.get("data") is None:
            return pd.DataFrame()
        df = pd.DataFrame(data["data"])
        return df

    def get_packing_yield_data(
        self,
        timestamp1: str = None,
        timestamp2: Optional[str] = None,
        interval: Optional[str] = None,
        operator: str = None,
    ):
        cube_client = CubeJSClient()
        query_params = {
            "dimensions": [
                "PACKING_YIELD.unit_name",
                "PACKING_YIELD.sku",
                "PACKING_YIELD.sale_order",
                "PACKING_YIELD.yield",
                "PACKING_YIELD.pak_wt",
                "PACKING_YIELD.updated_at",
            ],
            "order": {"PACKING_YIELD.created_at": "desc"},
            "filters": [{"member": "PACKING_YIELD.yield", "operator": "set"}],
        }
        params = self._build_date_filter(
            query_param=query_params,
            timestamp_column="PACKING_YIELD.updated_at",
            timestamp1=timestamp1,
            timestamp2=timestamp2,
            operator=operator,
        )
        cube_url = cube_client.get_base_url()
        params = {"query": json.dumps(params)}
        response = cube_client.api_call(
            url=cube_url, query_params=params, api_request_method="GET"
        )
        data = response.json()
        if not data or data.get("data") is None:
            return pd.DataFrame()
        df = pd.DataFrame(data["data"])
        return df

    def get_cooking_yield_data(
        self,
        timestamp1: str = None,
        timestamp2: Optional[str] = None,
        interval: Optional[str] = None,
        operator: str = None,
    ):
        cube_client = CubeJSClient()
        query_params = {
            "dimensions": [
                "COOKING.plant_lot_number",
                "COOKING.count_per_pound",
                "COOKING.sku",
                "COOKING.sale_order",
                "COOKING.time_minutes",
                "COOKING.temperature",
                "COOKING.unit_id",
            ],
            "measures": [
                "COOKING.cooking_temp",
                "COOKING.chilling_temp",
                "COOKING.water_zone_temp",
            ],
            "filters": [],
            "order": {"COOKING.plant_lot_number": "asc"},
        }
        params = self._build_date_filter(
            query_param=query_params,
            timestamp_column="COOKING.updated_at",
            timestamp1=timestamp1,
            timestamp2=timestamp2,
            operator=operator,
        )

        cube_url = cube_client.get_base_url()
        params = {"query": json.dumps(params)}

        response = cube_client.api_call(
            url=cube_url, query_params=params, api_request_method="GET"
        )

        data = response.json()
        if not data or data.get("data") is None:
            return pd.DataFrame()

        df = pd.DataFrame(data["data"])
        return df

    @staticmethod
    def _build_date_filter(
        query_param: dict,
        timestamp_column: str,
        timestamp1: Optional[str] = None,
        timestamp2: Optional[str] = None,
        operator: Optional[str] = None,
    ):
        if operator in ["inDateRange", "notInDateRange"] and timestamp1 and timestamp2:
            query_param["filters"].append(
                {
                    "member": timestamp_column,
                    "operator": operator,
                    "values": [timestamp1, timestamp2],
                }
            )
        elif timestamp1:
            if not operator or (operator == "equals" and len(timestamp1) <= 10):
                query_param["filters"].append(
                    {
                        "member": timestamp_column,
                        "operator": "inDateRange",
                        "values": [f"{timestamp1} 00:00:00", f"{timestamp1} 23:59:59"],
                    }
                )
            else:
                query_param["filters"].append(
                    {
                        "member": timestamp_column,
                        "operator": operator,
                        "values": [timestamp1],
                    }
                )
        elif timestamp2:
            query_param["filters"].append(
                {
                    "member": timestamp_column,
                    "operator": operator or "equals",
                    "values": [timestamp2],
                }
            )
        return query_param
