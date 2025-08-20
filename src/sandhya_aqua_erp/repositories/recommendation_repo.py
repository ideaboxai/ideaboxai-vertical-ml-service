import pandas as pd
from src.sandhya_aqua_erp.db_conn import get_sandhya_db_engine as get_db_connection

queries = {
    "grn_process_query": """SELECT gi.lot_number,
                gi.quantity as quantity_in_kg_at_GRN,
                gi.count as number_of_shrimps_in_a_kg_at_GRN,
                gi.vi_inspection as vi_inspection_during_GRN,
                gi.rm_inspection as rm_inspection_during_GRN,
                gi.antibiotic_test as antibiotic_test_during_GRN,
                gi.loose_shell as loose_shell_during_GRN,
                gi.fungus_percentage as fungus_percentage_during_GRN
            FROM erpx_dev_rm_procurement.grn_items gi
            WHERE gi.plant_lot_number = %s
""",
    "grading_process_query": """SELECT
                                g.lot_name,
                                g.crate_weight AS weight_of_each_crate,

                                /* Shrimp weight (total reading weight - total crate weight) */
                                (
                                    COALESCE(SUM(pr.weight), 0)
                                    - (g.crate_weight * COALESCE(SUM(pr.crates), 0))
                                ) AS weight_of_shrimp_in_kg_at_grading,

                                /* Combined grades + sensor readings per bin_number */
                                (
                                    SELECT COALESCE(
                                    JSON_ARRAYAGG(
                                        JSON_OBJECT(
                                        'bin_number',                                 b.bin_number,
                                        'uniformity_ratio',                           gg.uniformity_ratio,   -- from grades
                                        'number_of_shrimps_in_a_pound_at_grading',    r.readings_count       -- from readings
                                        )
                                    ),
                                    JSON_ARRAY()
                                    )
                                    FROM (
                                    /* all bin_numbers present in either table for this session */
                                    SELECT bin_number
                                    FROM erpx_dev_production.pp_grading_grades
                                    WHERE session_id = g.session_id

                                    UNION

                                    SELECT bin_number
                                    FROM erpx_dev_production.pp_grading_readings
                                    WHERE session_id = g.session_id
                                    ) AS b
                                    LEFT JOIN (
                                    /* one row per bin from grades */
                                    SELECT
                                        bin_number,
                                        `count`         AS grade_count,
                                        uniformity_ratio AS uniformity_ratio
                                    FROM erpx_dev_production.pp_grading_grades
                                    WHERE session_id = g.session_id
                                    GROUP BY bin_number
                                    ) AS gg
                                    ON gg.bin_number = b.bin_number
                                    LEFT JOIN (
                                    /* one row per bin from readings (sum counts across rows) */
                                    SELECT
                                        bin_number,
                                        `count` AS readings_count
                                    FROM erpx_dev_production.pp_grading_readings
                                    WHERE session_id = g.session_id
                                    GROUP BY bin_number
                                    ) AS r
                                    ON r.bin_number = b.bin_number
                                ) AS data_at_shrimp_grading_process

                                FROM erpx_dev_production.pp_grading AS g
                                LEFT JOIN erpx_dev_production.pp_grading_readings AS pr
                                ON pr.session_id = g.session_id
                                WHERE g.lot_name = %s
                                GROUP BY g.session_id, g.lot_name, g.crate_weight;

""",
    "soaking_process_query": """SELECT  psl.lot_number,
                                        psl.count AS "Number_of_shrimps_in_a_pound_before_soaking",
                                        p.soak_time AS "Threshold_Time_in_minutes_for_soaking_shrimps",
                                        TIMESTAMPDIFF(MINUTE, p.soak_started_at, p.closed_at) AS "Time_in_minutes_for_soaking_shrimps",
                                        p.tub_number As "Tub_in_which_the_lot_was_soaked",
                                        p.additives AS "Ingridients_of_Solution",
                                        p.sale_order As "Purchase_order_of_customer",
                                        p.sku AS "Stock_Keeping_Unit",
                                        p.process_type,
                                        p.status AS "Progress_of_Soaking",
                                        CEIL(SUM(psr.weight) - (SUM(psr.crates) * MAX(p.crate_weight))) AS soaking_weight
                                FROM erpx_dev_production.pp_soaking_lot psl
                                JOIN erpx_dev_production.pp_soaking p 
                                ON psl.session_id  = p.session_id
                                JOIN erpx_dev_production.pp_soaking_readings psr 
                                ON psl.session_id  = psr.session_id
                                where psl.lot_number = (SELECT DISTINCT lot_name from erpx_dev_production.pp_grading ppg where ppg.lot_name = %s)
                                AND psl.count in (select pgr.count FROM erpx_dev_production.pp_grading_readings pgr
                                        join erpx_dev_production.pp_grading pg 
                                        on pg.session_id  = pgr.session_id
                                        where pg.lot_name = %s
                                        group by pgr.count)
                                        GROUP BY psl.lot_number, psl.count;
""",
    "cooking_process_query": """SELECT
                                pcl.lot_number,
                                pcl.`count` AS "Number_of_shrimps_in_a_pound_before_cooking",
                                pcl.created_at         AS "lot_created_at",
                                pc.sale_order,
                                pc.sku,
                                pc.status AS "Status_of_Cooking",
                                pc.started_at AS "Start_Of_Cooking",
                                pc.closed_at AS "End_Of_Cooking",
                                pc.created_at          AS "cooking_created_at",
                                pc.cooking_temp AS "Temperature_that_should_be_at_the_time_of_Cooking_Shrimps",
                                pc.chilling_temp AS "Temperature_that_should_be_at_the_time_of_Cooling_Shrimps",

                /* readings: { "<location_name>": [ last 5 readings (newest first) ] } */
                (
                    SELECT COALESCE(
                    JSON_OBJECTAGG(per_loc.location_name, per_loc.readings_json),
                    JSON_OBJECT()
                    )
                    FROM (
                    SELECT
                        q.location_name,
                        CAST(
                        CONCAT(
                            '[',
                            IFNULL(GROUP_CONCAT(q.item ORDER BY q.epoch_date DESC SEPARATOR ','), ''),
                            ']'
                        ) AS JSON
                        ) AS readings_json
                    FROM (
                        /* pick last 5 per location for this session */
                        SELECT
                        r.location_name,
                        r.epoch_date,
                        JSON_OBJECT(
                            'epoch_date',           r.epoch_date,
                            'temperature',          r.temperature,
                            'temperature_type',     r.temperature_type,
                            'temperature_fault',    r.temperature_fault,
                            'periodic_check_fault', r.periodic_check_fault
                        ) AS item
                        FROM (
                        SELECT
                            pcr.*,
                            ROW_NUMBER() OVER (
                            PARTITION BY pcr.location_name
                            ORDER BY pcr.epoch_date DESC
                            ) AS rn
                        FROM erpx_dev_production.pp_cooking_readings pcr
                        WHERE pcr.session_id = pcl.session_id
                        ) AS r
                        WHERE r.rn <= 5
                    ) AS q
                    GROUP BY q.location_name
                    ) AS per_loc
                ) AS Last_5_Temperature_Readings_in_different_parts_of_cooking_process

                FROM erpx_dev_production.pp_cooking_lot AS pcl
                JOIN erpx_dev_production.pp_cooking     AS pc
                ON pc.session_id = pcl.session_id
                WHERE pcl.lot_number = (SELECT DISTINCT lot_name from erpx_dev_production.pp_grading ppg where ppg.lot_name = %s)
                AND pcl.count in (select pgr.count FROM erpx_dev_production.pp_grading_readings pgr
                        join erpx_dev_production.pp_grading pg 
                        on pg.session_id  = pgr.session_id
                        where pg.lot_name = %s
                        group by pgr.count)
""",
    "yield_calculation_query": """
                    WITH
                    hl_cte AS
                        (SELECT g.created_at GRN_Created_dt,lot_name, g.count as hon_count, g.quantity as hon_weight,p.created_at as Grading_Created_dt,
                            FLOOR(sum(v.weight)-(p.crate_weight*count(v.crates))) as hl_weight,
                            ((sum(v.weight)-(p.crate_weight*count(v.crates)))/g.quantity)*100 as Grading_yield
                        FROM erpx_dev_rm_procurement.grn_items g
                            LEFT join erpx_dev_production.pp_grading p
                            on g.plant_lot_number = p.lot_name
                            LEFT join erpx_dev_production.pp_grading_readings v
                            on p.session_id = v.session_id
                            group by g.created_at ,lot_name, g.count, g.quantity ,p.created_at ),
                    grading_cte AS (
                        SELECT
                            p.lot_name,
                            pgr.count grading_count,
                            CEIL(SUM(pgr.weight)) AS grading_ct_wt,
                            MAX(p.crate_weight) * SUM(pgr.crates) AS crate_weight,
                            CEIL(SUM(pgr.weight)) - (MAX(p.crate_weight) * SUM(pgr.crates)) AS grading_weight
                        FROM erpx_dev_production.pp_grading_readings pgr
                        JOIN erpx_dev_production.pp_grading p ON pgr.session_id = p.session_id
                        GROUP BY p.lot_name, pgr.count
                    ),
                    soaking_cte AS (
                        SELECT
                            sl.lot_number,
                            sl.count as soaking_cnt,
                            ps.sale_order as soaking_sale_order,
                            CEIL(SUM(p.weight) - (SUM(p.crates) * MAX(ps.crate_weight))) AS soaking_weight
                        FROM erpx_dev_production.pp_soaking_readings p
                        JOIN erpx_dev_production.pp_soaking_lot sl ON p.session_id = sl.session_id
                        JOIN erpx_dev_production.pp_soaking ps ON p.session_id = ps.session_id
                        GROUP BY sl.lot_number, sl.count, ps.sale_order
                    )
                    SELECT DATE_FORMAT(hl.GRN_Created_dt, '%%Y-%%m-%%d %%H:%%i') as GRN_Created_dt,hl.lot_name,hl.hon_count,hl.hon_weight,hl.Grading_Created_dt,hl.hl_weight, ROUND(hl.Grading_yield,2),
                    g.Grading_count,g.Grading_ct_wt,g.Grading_weight,s.Soaking_cnt,s.Soaking_weight,  
                        ROUND((s.soaking_weight / g.grading_weight) * 100,2) AS Peeling_yield
                    FROM hl_cte hl
                    LEFT JOIN grading_cte g ON hl.lot_name = g.lot_name
                    LEFT JOIN soaking_cte s ON g.lot_name = s.lot_number AND g.grading_count = s.soaking_cnt
                    where g.lot_name  = %s
                    order by hl.Grading_Created_dt;
""",
    "packing_yield_query": """select soak.unit_id,soak.sale_order soak_sale_order,pack.customer_po pack_sale_order,
                        soak.brand, soak.sku
                        -- ,pack.unit_id
                        -- ,pack.packing_style
                        -- ,pack.wt
                    ,
                        soak.weight soak_weight, pack.pak_wt, pack.pouch_qty, pack.mc_qty ,
                        round(100 * pack.pak_wt / soak.weight, 1) pack_yield
                    from
                        (
                        select
                            -- timestampdiff(MINUTE,ps.soak_started_at,ps.closed_at) soaked_time, ps.soak_time
                            -- ,timestampdiff(MINUTE,ps.soak_started_at,ps.closed_at) - ps.soak_time over_soak_time,
                            ps.unit_id
                    ,
                            ps.sku
                            -- ,ps.process_type
                    ,
                            group_concat(distinct ps.sale_order) sale_order
                    ,
                            group_concat(distinct ps.brand) brand
                    ,
                            round(sum(sr.weight - (ps.crate_weight * sr.crates)), 0) weight
                        from
                            erpx_dev_production.pp_soaking_lot sl
                        left join erpx_dev_production.pp_soaking_readings sr
                    on
                            sl.session_id = sr.session_id
                        left join erpx_dev_production.pp_soaking ps
                    on
                            sl.session_id = ps.session_id
                    -- 	where
                    -- 		closed_AT between '2025-07-30 00:00:00.100' and '2025-07-30 23:59:59.000'
                        group by
                            ps.unit_id,
                            ps.sku
                            -- ,ps.sale_order,ps.brand,ps.process_type
                        order by
                            1,
                            2,
                            3,
                            4
                            -- (soaked_time - soak_time > 120) or
                            -- soaked_time > 240
                    ) soak
                    left join
                    (
                        SELECT
                            pa.unit_id,
                            pa.product_sku,
                            group_concat(distinct ifnull(nullif(ltrim(pa.customer_po), ''), 'DUMMY')) customer_po,
                            -- pa.packing_style,
                            sum(substr(pa.packing_style, 1, 2) * net_weight) wt,
                            -- case when pa.unit_id = 'NlSqOgl7yGzFB2eLAADPl' then 'unit2' else 'unit1' end plant,
                            -- pa.no_of_bags,
                            sum(pouch.pouch_qty) pouch_qty,
                            sum(ma.mc_qty) mc_qty ,
                            sum(round(ma.mc_qty * substr(pa.packing_style, 1, 2) * net_weight)) pak_wt
                        FROM
                            erpx_dev_production.pp_packing pa
                        inner join ( /* MC Qty - Master Carton Quantity number of packets processed for specific unit, session line etc.. */
                            select session_id, unit_id , count(sequence_number) as mc_qty
                            from erpx_dev_production.pp_packing_master_readings
                            group by session_id, unit_id) ma
                        on pa.session_id = ma.session_id and pa.unit_id = ma.unit_id
                        inner join (/* Pouch Qty - Pouch Quantity number of pouches processed for specific SKU, session, line etc.. */
                            select session_id, unit_id, count(id) pouch_qty
                            from erpx_dev_production.pp_packing_reading
                            group by session_id, unit_id) pouch
                        on ma.session_id = pouch.session_id and ma.unit_id = pouch.unit_id
                        group by 1, 2
                        order by pa.customer_po, pa.unit_id
                    ) pack
                    on
                        soak.unit_id = pack.unit_id
                        -- and soak.sale_order = case when trim(pack.customer_po) ='' then 'DUMMY'  else pack.customer_po end
                        and pack.product_sku = soak.sku
                        WHERE (pack.unit_id, pack.product_sku) IN (
                    SELECT unit_id, sku from pp_soaking ps join pp_soaking_lot psl on psl.session_id = ps.session_id WHERE psl.lot_number = %s)
                    order by 1, 5, 2, 3
""",
}


async def run_predefined_query(query_key, params=None):
    if query_key not in queries:
        raise ValueError(f"Query key '{query_key}' not found.")
    sql = queries[query_key]

    engine = get_db_connection(connection_mode="async")
    async with engine.connect() as conn:
        df = await conn.run_sync(
            lambda sync_conn: pd.read_sql(sql, sync_conn, params=params)
        )
    return df


if __name__ == "__main__":
    df = run_predefined_query("packing_yield_query", params=("5P2-464",))
    df.to_json("new.json", index=False)
    print(df.head(n=5))
