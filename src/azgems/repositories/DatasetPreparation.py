from src.azgems.data_source_connection import get_clickhouse_client, get_cubejs_client
import pandas as pd
from typing import Literal


class DatasetPreparation:
    def __init__(
        self,
        customer_name="Walmart",
        threshold=5,
        start_timestamp=None,
        end_timestamp=None,
    ):
        self.customer_name = customer_name
        self.threshold_for_delay = threshold
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.client = get_cubejs_client()

    def get_necessary_dataset_for_training_from_cubejs(self) -> pd.DataFrame:
        shipment_dataset_query_1_cube = """
            {{
            "dimensions": [
                "SHIPMENT_ROW_FEATURES.bill_id",
                "SHIPMENT_ROW_FEATURES.bill_id_dup",
                "SHIPMENT_ROW_FEATURES.bni_created_time",
                "SHIPMENT_ROW_FEATURES.coo",
                "SHIPMENT_ROW_FEATURES.days_since_ship_so_far",
                "SHIPMENT_ROW_FEATURES.days_until_eta",
                "SHIPMENT_ROW_FEATURES.delay_days",
                "SHIPMENT_ROW_FEATURES.delivery_terms",
                "SHIPMENT_ROW_FEATURES.eta_dt",
                "SHIPMENT_ROW_FEATURES.item_brand",
                "SHIPMENT_ROW_FEATURES.item_manufacturer",
                "SHIPMENT_ROW_FEATURES.item_product_category",
                "SHIPMENT_ROW_FEATURES.item_size",
                "SHIPMENT_ROW_FEATURES.item_sku",
                "SHIPMENT_ROW_FEATURES.lead_time_days",
                "SHIPMENT_ROW_FEATURES.ocean_freight",
                "SHIPMENT_ROW_FEATURES.po_date_dt",
                "SHIPMENT_ROW_FEATURES.po_shipment_terms",
                "SHIPMENT_ROW_FEATURES.promised_transit_days",
                "SHIPMENT_ROW_FEATURES.quantity_in",
                "SHIPMENT_ROW_FEATURES.receipt_dt",
                "SHIPMENT_ROW_FEATURES.scac",
                "SHIPMENT_ROW_FEATURES.shipment_days",
                "SHIPMENT_ROW_FEATURES.shipped_dt",
                "SHIPMENT_ROW_FEATURES.tariff_amount",
                "SHIPMENT_ROW_FEATURES.tariff_type",
                "SHIPMENT_ROW_FEATURES.total_bcy",
                "SHIPMENT_ROW_FEATURES.vendor_id",
                "SHIPMENT_ROW_FEATURES.vendor_name"
            ],
            "timeDimensions": [],
            "filters": [
                {{
                "values": ["{customer_name}"],
                "member": "SHIPMENT_ROW_FEATURES.customer_name",
                "operator": "contains"
                }}
            ]
            }}
            """.format(
            customer_name="Walmart"
        )

        shipment_dataset_query_2_cube = """{{
            "dimensions": [
                "SHIPMENT_VENDOR_AGGREGATES.bill_id",
                "SHIPMENT_VENDOR_AGGREGATES.vendor_avg_promised_transit_days",
                "SHIPMENT_VENDOR_AGGREGATES.vendor_avg_realized_delay_days",
                "SHIPMENT_VENDOR_AGGREGATES.vendor_id",
                "SHIPMENT_VENDOR_AGGREGATES.vendor_on_time_rate",
                "SHIPMENT_VENDOR_AGGREGATES.vendor_p50_promised_transit_days",
                "SHIPMENT_VENDOR_AGGREGATES.vendor_p50_realized_delay_days",
                "SHIPMENT_VENDOR_AGGREGATES.vendor_p90_promised_transit_days",
                "SHIPMENT_VENDOR_AGGREGATES.vendor_p90_realized_delay_days",
                "SHIPMENT_VENDOR_AGGREGATES.vendor_shipments_with_receipt"
            ],
            "filters": [
                {{
                "values": [
                    "{customer_name}"
                ],
                "member": "SHIPMENT_VENDOR_AGGREGATES.customer_name",
                "operator": "contains"
                }}
            ]
            }}
            """.format(
            customer_name="Walmart"
        )

        result1 = self.client.api_call(
            self.client.get_base_url() + "?query=" + shipment_dataset_query_1_cube,
            "GET",
        )
        shipment_dataset_df1 = pd.DataFrame(result1.json()["data"])

        # Strip cube name prefix
        shipment_dataset_df1.columns = [
            col.split(".")[-1] for col in shipment_dataset_df1.columns
        ]

        result2 = self.client.api_call(
            self.client.get_base_url() + "?query=" + shipment_dataset_query_2_cube,
            "GET",
        )
        shipment_dataset_df2 = pd.DataFrame(result2.json()["data"])

        # Strip cube name prefix
        shipment_dataset_df2.columns = [
            col.split(".")[-1] for col in shipment_dataset_df2.columns
        ]

        shipment_dataset_df = shipment_dataset_df1.merge(
            shipment_dataset_df2, on=["bill_id", "vendor_id"], how="left"
        )

        return shipment_dataset_df

    def get_necessary_dataset_for_inference_from_cubejs(self) -> pd.DataFrame:
        inference_cube_query_1 = """ 
            {{
            "dimensions": [
                "SHIPMENT_ROW_FEATURES.bill_id",
                "SHIPMENT_ROW_FEATURES.bill_id_dup",
                "SHIPMENT_ROW_FEATURES.bni_created_time",
                "SHIPMENT_ROW_FEATURES.coo",
                "SHIPMENT_ROW_FEATURES.days_since_ship_so_far",
                "SHIPMENT_ROW_FEATURES.days_until_eta",
                "SHIPMENT_ROW_FEATURES.delay_days",
                "SHIPMENT_ROW_FEATURES.delivery_terms",
                "SHIPMENT_ROW_FEATURES.eta_dt",
                "SHIPMENT_ROW_FEATURES.item_brand",
                "SHIPMENT_ROW_FEATURES.item_manufacturer",
                "SHIPMENT_ROW_FEATURES.item_product_category",
                "SHIPMENT_ROW_FEATURES.item_size",
                "SHIPMENT_ROW_FEATURES.item_sku",
                "SHIPMENT_ROW_FEATURES.lead_time_days",
                "SHIPMENT_ROW_FEATURES.ocean_freight",
                "SHIPMENT_ROW_FEATURES.po_date_dt",
                "SHIPMENT_ROW_FEATURES.po_shipment_terms",
                "SHIPMENT_ROW_FEATURES.promised_transit_days",
                "SHIPMENT_ROW_FEATURES.quantity_in",
                "SHIPMENT_ROW_FEATURES.receipt_dt",
                "SHIPMENT_ROW_FEATURES.scac",
                "SHIPMENT_ROW_FEATURES.shipment_days",
                "SHIPMENT_ROW_FEATURES.shipped_dt",
                "SHIPMENT_ROW_FEATURES.tariff_amount",
                "SHIPMENT_ROW_FEATURES.tariff_type",
                "SHIPMENT_ROW_FEATURES.total_bcy",
                "SHIPMENT_ROW_FEATURES.vendor_id",
                "SHIPMENT_ROW_FEATURES.vendor_name"
            ],
            "timeDimensions": [],
            "filters": [
                {{
                "values": [
                    "{customer_name}"
                ],
                "member": "SHIPMENT_ROW_FEATURES.customer_name",
                "operator": "contains"
            }},
                {{
                "values": [
                    "{start_timestamp}",
                    "{end_timestamp}"
                ],
                "member": "SHIPMENT_ROW_FEATURES.bni_created_time",
                "operator": "inDateRange"
                }}
            ]
            }}
            """.format(
            customer_name=self.customer_name,
            start_timestamp=self.start_timestamp,
            end_timestamp=self.end_timestamp,
        )

        inference_cube_query_2 = """ {{
            "dimensions": [
                "SHIPMENT_VENDOR_AGGREGATES.bill_id",
                "SHIPMENT_VENDOR_AGGREGATES.customer_name",
                "SHIPMENT_VENDOR_AGGREGATES.vendor_avg_promised_transit_days",
                "SHIPMENT_VENDOR_AGGREGATES.vendor_avg_realized_delay_days",
                "SHIPMENT_VENDOR_AGGREGATES.vendor_id",
                "SHIPMENT_VENDOR_AGGREGATES.vendor_on_time_rate",
                "SHIPMENT_VENDOR_AGGREGATES.vendor_p50_promised_transit_days",
                "SHIPMENT_VENDOR_AGGREGATES.vendor_p50_realized_delay_days",
                "SHIPMENT_VENDOR_AGGREGATES.vendor_p90_promised_transit_days",
                "SHIPMENT_VENDOR_AGGREGATES.vendor_p90_realized_delay_days",
                "SHIPMENT_VENDOR_AGGREGATES.vendor_shipments_with_receipt"
            ],
            "filters": [
                {{
                "values": [
                    "{customer_name}"
                ],
                "member": "SHIPMENT_VENDOR_AGGREGATES.customer_name",
                "operator": "contains"
                }}
            ]
            }}""".format(
            customer_name=self.customer_name,
        )
        inference_result = self.client.api_call(
            self.client.get_base_url() + "?query=" + inference_cube_query_1, "GET"
        )
        inference_dataset_df1 = pd.DataFrame(inference_result.json()["data"])
        # Strip cube name prefix
        inference_dataset_df1.columns = [
            col.split(".")[-1] for col in inference_dataset_df1.columns
        ]

        inference_result = self.client.api_call(
            self.client.get_base_url() + "?query=" + inference_cube_query_2, "GET"
        )
        inference_dataset_df2 = pd.DataFrame(inference_result.json()["data"])
        # Strip cube name prefix
        inference_dataset_df2.columns = [
            col.split(".")[-1] for col in inference_dataset_df2.columns
        ]

        inference_dataset_df = inference_dataset_df1.merge(
            inference_dataset_df2, on=["bill_id", "vendor_id"], how="left"
        )

        return inference_dataset_df

    def get_necessary_dataset_for_training(self) -> pd.DataFrame:
        shipment_dataset_query = """
            SELECT
            -- Primary keys / bookkeeping
            bni.bill_id                                         AS bill_id,
            b.bill_id                                           AS bill_id_dup,
            v.vendor_id                                         AS vendor_id,
            v.vendor_name                                       AS vendor_name,
            bni.created_time                                    AS bni_created_time,
            

            -- 📦 Core dates (parsed) — receipt_date is intentionally NOT used for row-level features
            round(GREATEST(
                dateDiff('day', 
                    toDate(parseDateTimeBestEffortOrNull(b.eta)), 
                    toDate(parseDateTimeBestEffortOrNull(b.receipt_date))
                ), 
            0), 2) AS delay_days,
            parseDateTimeBestEffortOrNull(b.shipped_date)      AS shipped_dt,
            parseDateTimeBestEffortOrNull(b.eta)               AS eta_dt,
            parseDateTimeBestEffortOrNull(p.purchase_order_date) AS po_date_dt,
            parseDateTimeBestEffortOrNull(b.receipt_date)       AS receipt_dt,

            -- ⏩ Scoring-safe timing features (all available at ship-time / scoring-time)
            toFloat64(dateDiff('day',
                toDate(shipped_dt),
                toDate(ifNull(receipt_dt, eta_dt))              -- receipt if present, else ETA
            )) AS shipment_days,
            toFloat64(dateDiff('day', toDate(shipped_dt), toDate(eta_dt)))        AS promised_transit_days,
            /* days from now -> eta (negative if ETA already passed). 
            In offline training simulate `now()` with your scoring snapshot time. */
            toFloat64(dateDiff('day', toDate(now()), toDate(eta_dt)))            AS days_until_eta,
            /* how many days since shipped (so far) at scoring time */
            toFloat64(dateDiff('day', toDate(shipped_dt), toDate(now())))        AS days_since_ship_so_far,
            toFloat64(dateDiff('day', toDate(po_date_dt), toDate(shipped_dt)))   AS lead_time_days,


            -- 🌍 Shipping details (static/manifest fields)
            b.coo                                                                 AS coo,
            b.scac                                                                AS scac,
            round(toFloat64OrNull(b.tariff_amount), 2)                            AS tariff_amount,
            round(toFloat64OrNull(b.ocean_freight), 2)                            AS ocean_freight,
            p.delivery_terms                                                      AS delivery_terms,
            p.shipment_terms                                                      AS po_shipment_terms,
            p.tariff_type                                                         AS tariff_type,

            -- 💰 Cost metrics / product / vendor info
            p.total_bcy                                                           AS total_bcy,
            round(toFloat64OrNull(bni.quantity_in), 2)                            AS quantity_in,
            i.sku                                                                 AS item_sku,
            i.brand                                                               AS item_brand,
            i.manufacturer                                                        AS item_manufacturer,
            i.product_category                                                    AS item_product_category,
            i.size                                                                AS item_size,

            -- 🧮 Vendor historical aggregates (safe: computed only from prior shipments)
            -- Promised transit stats (ship -> eta) computed over trailing window prior to current shipped_dt
            round(vs.vendor_avg_promised_transit_days, 2)                         AS vendor_avg_promised_transit_days,
            round(vs.vendor_p50_promised_transit_days, 2)                         AS vendor_p50_promised_transit_days,
            round(vs.vendor_p90_promised_transit_days, 2)                         AS vendor_p90_promised_transit_days,

            -- Realized delay stats (eta -> receipt) computed only from historical rows with receipt_date
            -- NOTE: these are historical aggregates and safe (they do not include the current shipment)
            round(vs.vendor_avg_realized_delay_days, 2)                           AS vendor_avg_realized_delay_days,
            round(vs.vendor_p50_realized_delay_days, 2)                           AS vendor_p50_realized_delay_days,
            round(vs.vendor_p90_realized_delay_days, 2)                           AS vendor_p90_realized_delay_days,
            round(vs.vendor_on_time_rate, 4)                                      AS vendor_on_time_rate,
            vs.vendor_shipments_with_receipt                                      AS vendor_shipments_with_receipt

        FROM zoho_books_analytics.batch_number_in AS bni
        INNER JOIN zoho_books_analytics.bills AS b
            ON bni.bill_id = b.bill_id
        INNER JOIN zoho_books_analytics.bill_item AS bi
            ON b.bill_id = bi.bill_id
        INNER JOIN zoho_books_analytics.purchase_orders AS p
            ON b.purchase_order = p.purchase_order_number
        INNER JOIN zoho_books_analytics.items AS i
            ON bi.product_id = i.item_id
        INNER JOIN zoho_books_analytics.sales_orders AS so
            ON p.reference_number = so.sales_order
        INNER JOIN zoho_books_analytics.customers AS c
            ON c.customer_id = so.customer_id
        INNER JOIN zoho_books_analytics.customer_item_mapping AS ci
            ON i.sku = ci.az_sku
        INNER JOIN zoho_books_analytics.vendors AS v
            ON v.vendor_id = b.vendor_id

        /* Safe per-bill vendor aggregates: trailing 365-day window, strictly before current shipped_dt */
        LEFT JOIN
        (
            /* For each current bill, aggregate prior shipments from same vendor
            within the 365 days before the current shipped_date (exclude current day). */
            SELECT
                b_curr.bill_id                                                         AS bill_id_key,
                v_curr.vendor_id                                                       AS vendor_id,

                -- Promised transit (ship -> eta) historical metrics
                AVG(toFloat64(dateDiff('day',
                    toDate(parseDateTimeBestEffortOrNull(b_hist.shipped_date)),
                    toDate(parseDateTimeBestEffortOrNull(b_hist.eta))
                )))                                                                      AS vendor_avg_promised_transit_days,

                quantileExact(0.5)(toFloat64(dateDiff('day',
                    toDate(parseDateTimeBestEffortOrNull(b_hist.shipped_date)),
                    toDate(parseDateTimeBestEffortOrNull(b_hist.eta))
                )))                                                                      AS vendor_p50_promised_transit_days,

                quantileExact(0.9)(toFloat64(dateDiff('day',
                    toDate(parseDateTimeBestEffortOrNull(b_hist.shipped_date)),
                    toDate(parseDateTimeBestEffortOrNull(b_hist.eta))
                )))                                                                      AS vendor_p90_promised_transit_days,

                -- Realized delay (eta -> receipt) historical metrics: include only history rows with receipt_date
                AVG(toFloat64(greatest(
                    dateDiff('day',
                        toDate(parseDateTimeBestEffortOrNull(b_hist.eta)),
                        toDate(parseDateTimeBestEffortOrNull(b_hist.receipt_date))
                    ), 0
                )))                                                                      AS vendor_avg_realized_delay_days,

                quantileExact(0.5)(toFloat64(greatest(
                    dateDiff('day',
                        toDate(parseDateTimeBestEffortOrNull(b_hist.eta)),
                        toDate(parseDateTimeBestEffortOrNull(b_hist.receipt_date))
                    ), 0
                )))                                                                      AS vendor_p50_realized_delay_days,

                quantileExact(0.9)(toFloat64(greatest(
                    dateDiff('day',
                        toDate(parseDateTimeBestEffortOrNull(b_hist.eta)),
                        toDate(parseDateTimeBestEffortOrNull(b_hist.receipt_date))
                    ), 0
                )))                                                                      AS vendor_p90_realized_delay_days,

                -- On-time rate among historical rows (eta->receipt == 0)
                AVG(toUInt8(greatest(
                    dateDiff('day',
                        toDate(parseDateTimeBestEffortOrNull(b_hist.eta)),
                        toDate(parseDateTimeBestEffortOrNull(b_hist.receipt_date))
                    ), 0) = 0
                ))                                                                      AS vendor_on_time_rate,

                -- Count of usable historical shipments in the window
                COUNT()                                                                  AS vendor_shipments_with_receipt

            FROM zoho_books_analytics.bills AS b_curr
            INNER JOIN zoho_books_analytics.vendors AS v_curr
                ON v_curr.vendor_id = b_curr.vendor_id

            /* historical shipments for the vendor */
            INNER JOIN zoho_books_analytics.bills AS b_hist
                ON b_hist.vendor_id = v_curr.vendor_id
            AND b_hist.shipped_date IS NOT NULL
            AND b_hist.eta IS NOT NULL
            AND b_hist.receipt_date IS NOT NULL
            AND b_hist.receipt_date != ''
            /* strictly before current shipped_date, within trailing 365 days */
            AND toDate(parseDateTimeBestEffortOrNull(b_hist.shipped_date))
                BETWEEN addDays(toDate(parseDateTimeBestEffortOrNull(b_curr.shipped_date)), -365)
                    AND addDays(toDate(parseDateTimeBestEffortOrNull(b_curr.shipped_date)), -1)

            /* keep same customer scope as main query */
            INNER JOIN zoho_books_analytics.purchase_orders AS p_hist
                ON b_hist.purchase_order = p_hist.purchase_order_number
            INNER JOIN zoho_books_analytics.sales_orders AS so_hist
                ON p_hist.reference_number = so_hist.sales_order
            INNER JOIN zoho_books_analytics.customers AS c_hist
                ON c_hist.customer_id = so_hist.customer_id

            WHERE
                c_hist.customer_name LIKE 'Walmart%'
                AND b_curr.shipped_date IS NOT NULL
                AND b_curr.eta IS NOT NULL

            GROUP BY
                b_curr.bill_id, v_curr.vendor_id
        ) AS vs
            ON vs.bill_id_key = b.bill_id
        AND vs.vendor_id   = v.vendor_id

        WHERE
            c.customer_name LIKE 'Walmart%'
            AND b.shipped_date IS NOT NULL
            AND b.eta IS NOT NULL

        ORDER BY
            bni.created_time DESC
            """
        client = get_clickhouse_client()
        # try:
        #     result = client.api_call(
        #         client.get_base_url() + "?query=" + query_for_cubejs, "GET"
        #     )
        #     dataset_df = pd.DataFrame(result.json()["data"])

        #     # Strip cube name prefix
        #     dataset_df.columns = [col.split(".")[-1] for col in dataset_df.columns]
        #     return dataset_df
        # except Exception as e:
        #     print(e)
        #     raise Exception(f"Error getting necessary dataset for training: {e}")
        result = client.query(shipment_dataset_query)

        shipment_dataset_df = pd.DataFrame(
            result.result_rows, columns=[col for col in result.column_names]
        )

        return shipment_dataset_df

    def clean_dataset(
        self, method: Literal["training", "inference"] = "training"
    ) -> pd.DataFrame:
        dataset = (
            self.get_necessary_dataset_for_training_from_cubejs()
            if method == "training"
            else self.get_necessary_dataset_for_inference_from_cubejs()
        )

        dataset = dataset.fillna({"tariff_amount": 0, "ocean_freight": 0})
        dataset = dataset.replace({"tariff_type": {"": "Not Applicable"}})
        dataset = dataset.drop_duplicates()

        date_time_columns = ["shipped_dt", "eta_dt", "receipt_dt"]
        dataset[date_time_columns] = dataset[date_time_columns].apply(
            pd.to_datetime, errors="coerce"
        )

        shipped_dt = dataset["shipped_dt"]
        dataset["shipped_date_weekday"] = shipped_dt.dt.weekday
        dataset["shipped_date_month"] = shipped_dt.dt.month
        dataset["shipped_date_day"] = shipped_dt.dt.day

        dataset["total_bcy"] = (
            dataset["total_bcy"]
            .astype(str)
            .str.replace("USD", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
            .replace("", "0")
            .astype(float)
        )

        numeric_cols = [
            "delay_days",
            "shipment_days",
            "lead_time_days",
            "quantity_in",
            "total_bcy",
            "ocean_freight",
            "tariff_amount",
            "promised_transit_days",
            "lead_time_days",
            "days_since_ship_so_far",
            # "days_until_eta",
            "vendor_avg_promised_transit_days",
            "vendor_p50_promised_transit_days",
            "vendor_p90_promised_transit_days",
            "vendor_avg_realized_delay_days",
            "vendor_p50_realized_delay_days",
            "vendor_p90_realized_delay_days",
            "vendor_on_time_rate",
            "vendor_shipments_with_receipt",
        ]

        for col in numeric_cols:
            dataset[col] = pd.to_numeric(dataset[col], errors="coerce")
            dataset[col] = dataset[col].where(dataset[col] >= 0)

        distance_dict = {
            "INDIA": 11000,
            "CHINA": 6000,
            "INDONESIA": 8200,
            "VIETNAM": 6200,
            "ECUADOR": 2100,
            "THAILAND": 8100,
        }
        dataset["distance_nm"] = dataset["coo"].map(distance_dict)

        dataset = dataset.dropna() if method == "training" else dataset

        dataset = dataset.drop(
            columns=[
                "coo",
                "item_product_category",
                "bill_id",
                "bill_id_dup",
                "vendor_id",
                "bni_created_time",
                "po_date_dt",
                "receipt_dt",
                "vendor_name",
            ]
            + date_time_columns
        )
        dataset.to_csv("dataset_verification.csv", index=False)
        return dataset

    def get_necessary_dataset_for_inference(self) -> pd.DataFrame:
        inference_data_query = """ 
                SELECT
                -- Primary keys / bookkeeping
                bni.bill_id                                         AS bill_id,
                b.bill_id                                           AS bill_id_dup,
                v.vendor_id                                         AS vendor_id,
                v.vendor_name                                       AS vendor_name,
                bni.created_time                                    AS bni_created_time,
                

                -- 📦 Core dates (parsed) — receipt_date is intentionally NOT used for row-level features
                round(GREATEST(
                    dateDiff('day', 
                        toDate(parseDateTimeBestEffortOrNull(b.eta)), 
                        toDate(parseDateTimeBestEffortOrNull(b.receipt_date))
                    ), 
                0), 2) AS delay_days,
                parseDateTimeBestEffortOrNull(b.shipped_date)      AS shipped_dt,
                parseDateTimeBestEffortOrNull(b.eta)               AS eta_dt,
                parseDateTimeBestEffortOrNull(p.purchase_order_date) AS po_date_dt,
                parseDateTimeBestEffortOrNull(b.receipt_date)       AS receipt_dt,

                -- ⏩ Scoring-safe timing features (all available at ship-time / scoring-time)
                toFloat64(dateDiff('day',
                    toDate(shipped_dt),
                    toDate(ifNull(receipt_dt, eta_dt))              -- receipt if present, else ETA
                )) AS shipment_days,
                toFloat64(dateDiff('day', toDate(shipped_dt), toDate(eta_dt)))        AS promised_transit_days,
                /* days from now -> eta (negative if ETA already passed). 
                In offline training simulate `now()` with your scoring snapshot time. */
                toFloat64(dateDiff('day', toDate(now()), toDate(eta_dt)))            AS days_until_eta,
                /* how many days since shipped (so far) at scoring time */
                toFloat64(dateDiff('day', toDate(shipped_dt), toDate(now())))        AS days_since_ship_so_far,
                toFloat64(dateDiff('day', toDate(po_date_dt), toDate(shipped_dt)))   AS lead_time_days,


                -- 🌍 Shipping details (static/manifest fields)
                b.coo                                                                 AS coo,
                b.scac                                                                AS scac,
                round(toFloat64OrNull(b.tariff_amount), 2)                            AS tariff_amount,
                round(toFloat64OrNull(b.ocean_freight), 2)                            AS ocean_freight,
                p.delivery_terms                                                      AS delivery_terms,
                p.shipment_terms                                                      AS po_shipment_terms,
                p.tariff_type                                                         AS tariff_type,

                -- 💰 Cost metrics / product / vendor info
                p.total_bcy                                                           AS total_bcy,
                round(toFloat64OrNull(bni.quantity_in), 2)                            AS quantity_in,
                i.sku                                                                 AS item_sku,
                i.brand                                                               AS item_brand,
                i.manufacturer                                                        AS item_manufacturer,
                i.product_category                                                    AS item_product_category,
                i.size                                                                AS item_size,

                -- 🧮 Vendor historical aggregates (safe: computed only from prior shipments)
                -- Promised transit stats (ship -> eta) computed over trailing window prior to current shipped_dt
                round(vs.vendor_avg_promised_transit_days, 2)                         AS vendor_avg_promised_transit_days,
                round(vs.vendor_p50_promised_transit_days, 2)                         AS vendor_p50_promised_transit_days,
                round(vs.vendor_p90_promised_transit_days, 2)                         AS vendor_p90_promised_transit_days,

                -- Realized delay stats (eta -> receipt) computed only from historical rows with receipt_date
                -- NOTE: these are historical aggregates and safe (they do not include the current shipment)
                round(vs.vendor_avg_realized_delay_days, 2)                           AS vendor_avg_realized_delay_days,
                round(vs.vendor_p50_realized_delay_days, 2)                           AS vendor_p50_realized_delay_days,
                round(vs.vendor_p90_realized_delay_days, 2)                           AS vendor_p90_realized_delay_days,
                round(vs.vendor_on_time_rate, 4)                                      AS vendor_on_time_rate,
                vs.vendor_shipments_with_receipt                                      AS vendor_shipments_with_receipt

            FROM zoho_books_analytics.batch_number_in AS bni
            INNER JOIN zoho_books_analytics.bills AS b
                ON bni.bill_id = b.bill_id
            INNER JOIN zoho_books_analytics.bill_item AS bi
                ON b.bill_id = bi.bill_id
            INNER JOIN zoho_books_analytics.purchase_orders AS p
                ON b.purchase_order = p.purchase_order_number
            INNER JOIN zoho_books_analytics.items AS i
                ON bi.product_id = i.item_id
            INNER JOIN zoho_books_analytics.sales_orders AS so
                ON p.reference_number = so.sales_order
            INNER JOIN zoho_books_analytics.customers AS c
                ON c.customer_id = so.customer_id
            INNER JOIN zoho_books_analytics.customer_item_mapping AS ci
                ON i.sku = ci.az_sku
            INNER JOIN zoho_books_analytics.vendors AS v
                ON v.vendor_id = b.vendor_id

            /* Safe per-bill vendor aggregates: trailing 365-day window, strictly before current shipped_dt */
            LEFT JOIN
            (
                /* For each current bill, aggregate prior shipments from same vendor
                within the 365 days before the current shipped_date (exclude current day). */
                SELECT
                    b_curr.bill_id                                                         AS bill_id_key,
                    v_curr.vendor_id                                                       AS vendor_id,

                    -- Promised transit (ship -> eta) historical metrics
                    AVG(toFloat64(dateDiff('day',
                        toDate(parseDateTimeBestEffortOrNull(b_hist.shipped_date)),
                        toDate(parseDateTimeBestEffortOrNull(b_hist.eta))
                    )))                                                                      AS vendor_avg_promised_transit_days,

                    quantileExact(0.5)(toFloat64(dateDiff('day',
                        toDate(parseDateTimeBestEffortOrNull(b_hist.shipped_date)),
                        toDate(parseDateTimeBestEffortOrNull(b_hist.eta))
                    )))                                                                      AS vendor_p50_promised_transit_days,

                    quantileExact(0.9)(toFloat64(dateDiff('day',
                        toDate(parseDateTimeBestEffortOrNull(b_hist.shipped_date)),
                        toDate(parseDateTimeBestEffortOrNull(b_hist.eta))
                    )))                                                                      AS vendor_p90_promised_transit_days,

                    -- Realized delay (eta -> receipt) historical metrics: include only history rows with receipt_date
                    AVG(toFloat64(greatest(
                        dateDiff('day',
                            toDate(parseDateTimeBestEffortOrNull(b_hist.eta)),
                            toDate(parseDateTimeBestEffortOrNull(b_hist.receipt_date))
                        ), 0
                    )))                                                                      AS vendor_avg_realized_delay_days,

                    quantileExact(0.5)(toFloat64(greatest(
                        dateDiff('day',
                            toDate(parseDateTimeBestEffortOrNull(b_hist.eta)),
                            toDate(parseDateTimeBestEffortOrNull(b_hist.receipt_date))
                        ), 0
                    )))                                                                      AS vendor_p50_realized_delay_days,

                    quantileExact(0.9)(toFloat64(greatest(
                        dateDiff('day',
                            toDate(parseDateTimeBestEffortOrNull(b_hist.eta)),
                            toDate(parseDateTimeBestEffortOrNull(b_hist.receipt_date))
                        ), 0
                    )))                                                                      AS vendor_p90_realized_delay_days,

                    -- On-time rate among historical rows (eta->receipt == 0)
                    AVG(toUInt8(greatest(
                        dateDiff('day',
                            toDate(parseDateTimeBestEffortOrNull(b_hist.eta)),
                            toDate(parseDateTimeBestEffortOrNull(b_hist.receipt_date))
                        ), 0) = 0
                    ))                                                                      AS vendor_on_time_rate,

                    -- Count of usable historical shipments in the window
                    COUNT()                                                                  AS vendor_shipments_with_receipt

                FROM zoho_books_analytics.bills AS b_curr
                INNER JOIN zoho_books_analytics.vendors AS v_curr
                    ON v_curr.vendor_id = b_curr.vendor_id

                /* historical shipments for the vendor */
                INNER JOIN zoho_books_analytics.bills AS b_hist
                    ON b_hist.vendor_id = v_curr.vendor_id
                AND b_hist.shipped_date IS NOT NULL
                AND b_hist.eta IS NOT NULL
                AND b_hist.receipt_date IS NOT NULL
                AND b_hist.receipt_date != ''
                /* strictly before current shipped_date, within trailing 365 days */
                AND toDate(parseDateTimeBestEffortOrNull(b_hist.shipped_date))
                    BETWEEN addDays(toDate(parseDateTimeBestEffortOrNull(b_curr.shipped_date)), -365)
                        AND addDays(toDate(parseDateTimeBestEffortOrNull(b_curr.shipped_date)), -1)

                /* keep same customer scope as main query */
                INNER JOIN zoho_books_analytics.purchase_orders AS p_hist
                    ON b_hist.purchase_order = p_hist.purchase_order_number
                INNER JOIN zoho_books_analytics.sales_orders AS so_hist
                    ON p_hist.reference_number = so_hist.sales_order
                INNER JOIN zoho_books_analytics.customers AS c_hist
                    ON c_hist.customer_id = so_hist.customer_id

                WHERE
                    c_hist.customer_name LIKE 'Walmart%'
                    AND b_curr.shipped_date IS NOT NULL
                    AND b_curr.eta IS NOT NULL

                GROUP BY
                    b_curr.bill_id, v_curr.vendor_id
            ) AS vs
                ON vs.bill_id_key = b.bill_id
            AND vs.vendor_id   = v.vendor_id

            WHERE
                c.customer_name LIKE 'Walmart%'
                AND bni.created_time BETWEEN '{start_timestamp}' AND '{end_timestamp}'
            ORDER BY bni.created_time DESC
                """  # need to pass the start_timestamp and end_timestamp
        inference_data_query = inference_data_query.format(
            start_timestamp=self.start_timestamp,
            end_timestamp=self.end_timestamp,
        )
        client = get_clickhouse_client()
        result = client.query(inference_data_query)

        inference_dataset_df = pd.DataFrame(
            result.result_rows, columns=[col for col in result.column_names]
        )
        return inference_dataset_df


if __name__ == "__main__":
    dataset_preparation = DatasetPreparation(
        start_timestamp="2025-08-01", end_timestamp="2025-09-08"
    )
    dataset = dataset_preparation.clean_dataset(
        method="training"
    )  # testing for the logic of training dataset
    print(dataset.head())
    print(dataset.shape)
