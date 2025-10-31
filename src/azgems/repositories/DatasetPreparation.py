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

    def get_necessary_dataset_for_training(self) -> pd.DataFrame:
        shipment_dataset_query = """
            SELECT 
                -- 📦 Core dates (converted)
                parseDateTimeBestEffortOrNull(b.shipped_date) AS shipped_date,
                parseDateTimeBestEffortOrNull(b.eta) AS eta,
                parseDateTimeBestEffortOrNull(b.receipt_date) AS receipt_date,

                -- ⏱ Derived timing features
                round(GREATEST(
                    dateDiff('day', 
                        toDate(parseDateTimeBestEffortOrNull(b.eta)), 
                        toDate(parseDateTimeBestEffortOrNull(b.receipt_date))
                    ), 
                0), 2) AS delay_days,

                round(dateDiff('day', 
                    toDate(parseDateTimeBestEffortOrNull(b.shipped_date)), 
                    toDate(parseDateTimeBestEffortOrNull(b.receipt_date))
                ), 2) AS shipping_duration_days,

                round(dateDiff('day', 
                    toDate(parseDateTimeBestEffortOrNull(p.purchase_order_date)), 
                    toDate(parseDateTimeBestEffortOrNull(b.shipped_date))
                ), 2) AS lead_time_days,

                (parseDateTimeBestEffortOrNull(b.receipt_date) < parseDateTimeBestEffortOrNull(b.eta)) AS is_early_delivery,

                -- 🌍 Shipping details
                b.coo AS coo,
                b.scac AS scac,
                round(toFloat64OrNull(b.tariff_amount), 2) AS tariff_amount,
                round(toFloat64OrNull(b.ocean_freight), 2) AS ocean_freight,
                p.delivery_terms AS delivery_terms,
                p.shipment_terms AS po_shipment_terms,
                p.tariff_type AS tariff_type,

                -- 💰 Cost metrics
                p.total_bcy AS total_bcy,

                -- 📊 Product info
                round(toFloat64OrNull(bni.quantity_in), 2) AS quantity_in,
                i.sku AS item_sku,
                i.brand AS item_brand,
                i.manufacturer AS item_manufacturer,
                i.product_category AS item_product_category,
                i.size AS item_size,

                -- 🏢 Vendor info
                v.vendor_name AS vendor_name,

                -- 🧮 🔁 Vendor-level aggregates (joined)
                round(vs.vendor_avg_delay_days, 2) AS vendor_avg_delay_days,
                vs.vendor_shipments,
                round(vs.vendor_on_time_rate, 2) AS vendor_on_time_rate,
                round(vs.vendor_p50_delay_days, 2) AS vendor_p50_delay_days,
                round(vs.vendor_p90_delay_days, 2) AS vendor_p90_delay_days

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

            /* ✅ Inline vendor-level aggregate */
            LEFT JOIN
            (
                SELECT
                    v.vendor_id AS vendor_id,
                    round(avg(GREATEST(
                            dateDiff('day',
                                toDate(parseDateTimeBestEffortOrNull(b.eta)),
                                toDate(parseDateTimeBestEffortOrNull(b.receipt_date))
                            ), 0)), 2) AS vendor_avg_delay_days,
                    count() AS vendor_shipments,
                    round(avg(GREATEST(
                            dateDiff('day',
                                toDate(parseDateTimeBestEffortOrNull(b.eta)),
                                toDate(parseDateTimeBestEffortOrNull(b.receipt_date))
                            ), 0) = 0), 2) AS vendor_on_time_rate,
                    round(quantileExact(0.5)(GREATEST(
                            dateDiff('day',
                                toDate(parseDateTimeBestEffortOrNull(b.eta)),
                                toDate(parseDateTimeBestEffortOrNull(b.receipt_date))
                            ), 0)), 2) AS vendor_p50_delay_days,
                    round(quantileExact(0.9)(GREATEST(
                            dateDiff('day',
                                toDate(parseDateTimeBestEffortOrNull(b.eta)),
                                toDate(parseDateTimeBestEffortOrNull(b.receipt_date))
                            ), 0)), 2) AS vendor_p90_delay_days
                FROM zoho_books_analytics.bills AS b
                INNER JOIN zoho_books_analytics.purchase_orders AS p
                    ON b.purchase_order = p.purchase_order_number
                INNER JOIN zoho_books_analytics.sales_orders AS so
                    ON p.reference_number = so.sales_order
                INNER JOIN zoho_books_analytics.customers AS c
                    ON c.customer_id = so.customer_id
                INNER JOIN zoho_books_analytics.vendors AS v
                    ON v.vendor_id = b.vendor_id
                WHERE
                    c.customer_name LIKE 'Walmart%'
                    AND b.receipt_date != ''
                    AND b.shipped_date IS NOT NULL
                GROUP BY v.vendor_id
            ) AS vs
                ON vs.vendor_id = v.vendor_id

            WHERE 
                c.customer_name LIKE 'Walmart%' 
                AND b.receipt_date != ''
                AND b.shipped_date IS NOT NULL
            ORDER BY bni.created_time DESC
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
            self.get_necessary_dataset_for_training()
            if method == "training"
            else self.get_necessary_dataset_for_inference()
        )

        dataset = dataset.fillna({"tariff_amount": 0, "ocean_freight": 0})
        dataset = dataset.replace({"tariff_type": {"": "Not Applicable"}})
        dataset = dataset.drop_duplicates()

        date_time_columns = ["shipped_date", "eta", "receipt_date"]
        dataset[date_time_columns] = dataset[date_time_columns].apply(
            pd.to_datetime, errors="coerce"
        )

        shipped_dt = dataset["shipped_date"]
        dataset["shipped_date_weekday"] = shipped_dt.dt.weekday
        dataset["shipped_date_month"] = shipped_dt.dt.month
        dataset["shipped_date_day"] = shipped_dt.dt.day

        dataset = dataset.drop(columns=date_time_columns)

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
            "shipping_duration_days",
            "lead_time_days",
            "quantity_in",
            "total_bcy",
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

        dataset = dataset.drop(columns=["coo", "item_product_category"])

        return dataset

    def get_necessary_dataset_for_inference(self) -> pd.DataFrame:
        return pd.DataFrame()

if __name__ == "__main__":
    dataset_preparation = DatasetPreparation()
    dataset = dataset_preparation.clean_dataset()   # testing for the logic of training dataset
    print(dataset.head())