from src.azgems.data_source_connection import get_clickhouse_client, get_cubejs_client
import pandas as pd
from typing import Literal


# def fill_unknown_for_object_cols(df: pd.DataFrame) -> pd.DataFrame:
#     for col in df.select_dtypes(include=["object"]).columns:
#         df[col] = df[col].replace(r"^\s*$", "unknown", regex=True)
#     return df


class DatasetPreparation:
    def __init__(
        self,
        customer_name="Walmart",
        po_comitted="Direct Sale",
        threshold=7,
        start_timestamp=None,
        end_timestamp=None,
    ):
        self.customer_name = customer_name
        self.po_comitted = po_comitted
        self.threshold_for_delay = threshold
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp

    def get_necessary_dataset_for_training(self) -> pd.DataFrame:
        # query_for_dataset = f"""
        #             SELECT
        #                 bni.quantity_in,

        #                 -- From bills
        #                 b.bill_date,
        #                 b.due_date,
        #                 b.eta AS eta,
        #                 b.seal,
        #                 b.customs_broker,
        #                 b.receipt_date,
        #                 b.ocean_freight,

        #                 -- From items
        #                 i.yield_percentage,

        #             FROM
        #                 zoho_books_analytics.batch_number_in bni
        #             JOIN
        #                 zoho_books_analytics.bills b
        #                 ON bni.bill_id = b.bill_id
        #             JOIN
        #                 zoho_books_analytics.bill_item bi
        #                 ON bi.bill_id = b.bill_id
        #             JOIN
        #                 zoho_books_analytics.purchase_orders po
        #                 ON po.purchase_order_number = b.purchase_order
        #             JOIN
        #                 zoho_books_analytics.items i
        #                 ON i.item_id = bi.product_id
        #             JOIN
        #                 zoho_books_analytics.customer_item_mapping cim
        #                 ON i.sku = cim.az_sku

        #             WHERE
        #                 cim.customer_name LIKE '{self.customer_name}'
        #                 AND po.po_commited != '{self.po_comitted}'

        #             ORDER BY bni.created_time DESC
        #     """
        # client = get_clickhouse_client()
        query_for_cubejs = """{{
                    "dimensions": [
                            "DATA_FOR_ML_SERVICES.bill_date",
                            "DATA_FOR_ML_SERVICES.brand",
                            "DATA_FOR_ML_SERVICES.coo",
                            "DATA_FOR_ML_SERVICES.due_date",
                            "DATA_FOR_ML_SERVICES.eta",
                            "DATA_FOR_ML_SERVICES.manufacturer",
                            "DATA_FOR_ML_SERVICES.payment_terms",
                            "DATA_FOR_ML_SERVICES.quantity_in",
                            "DATA_FOR_ML_SERVICES.receipt_date",
                            "DATA_FOR_ML_SERVICES.scac",
                            "DATA_FOR_ML_SERVICES.shipping_port",
                            "DATA_FOR_ML_SERVICES.sku",
                            "DATA_FOR_ML_SERVICES.tariff_amount",
                            "DATA_FOR_ML_SERVICES.shipped_date"
                    ],
                    "filters": [
                        {{
                            "values": ["{customer_name}"],
                            "member": "DATA_FOR_ML_SERVICES.customer_name",
                            "operator": "contains"
                        }}
                    ],
                    "measures": []
                }}""".format(
            customer_name=self.customer_name
        )

        client = get_cubejs_client()
        try:
            result = client.api_call(
                client.get_base_url() + "?query=" + query_for_cubejs, "GET"
            )
            dataset_df = pd.DataFrame(result.json()["data"])

            # Strip cube name prefix
            dataset_df.columns = [col.split(".")[-1] for col in dataset_df.columns]
            return dataset_df
        except Exception as e:
            print(e)
            raise Exception(f"Error getting necessary dataset for training: {e}")

    def clean_dataset(
        self, task: Literal["training", "inference"] = "training"
    ) -> pd.DataFrame:
        dataset = (
            self.get_necessary_dataset_for_training()
            if task == "training"
            else self.get_necessary_dataset_for_inference()
        )
        if dataset.empty:
            raise Exception(f"No data available for {task}.")
        # need to add other logics for cleaning
        print("Initial dataset shape:", dataset.shape)
        print("The number of duplicate rows:", dataset.duplicated().sum())
        dataset.drop_duplicates(inplace=True)
        # Replace empty strings or whitespace-only values with NaN first
        dataset["shipping_port"].replace(r"^\s*$", pd.NA, regex=True, inplace=True)
        dataset["tariff_amount"].replace(r"^\s*$", pd.NA, regex=True, inplace=True)
        # dataset["customs_broker"].replace(r"^\s*$", pd.NA, regex=True, inplace=True)
        dataset["receipt_date"].replace(r"^\s*$", pd.NA, regex=True, inplace=True)

        # Then fill both NaN and now-empty ones
        dataset["shipping_port"].fillna("Unknown", inplace=True)
        dataset["tariff_amount"].fillna(0, inplace=True)
        # dataset["customs_broker"].fillna("Unknown", inplace=True)

        dataset["quantity_in"] = dataset["quantity_in"].astype(float)
        dataset["tariff_amount"] = dataset["tariff_amount"].astype(float)
        dataset["sku"] = dataset["sku"].astype(float)
        return dataset

    def calculate_target_variable_from_clean_dataset(
        self, task: Literal["training", "inference"] = "training"
    ):
        cleaned_df = self.clean_dataset(task=task)
        cleaned_df = cleaned_df.dropna(subset=["receipt_date", "eta"]) if task == "training" else cleaned_df
        # converting date columns to datetime
        date_columns = ["bill_date", "due_date", "eta", "shipped_date", "receipt_date"]
        for col in date_columns:
            cleaned_df[col] = pd.to_datetime(
                cleaned_df[col], format="%d %b %Y", errors="coerce"
            )

        cleaned_df["estimated_travel_duration"] = (
            cleaned_df["eta"] - cleaned_df["shipped_date"]
        ).dt.days
        cleaned_df["payment_window"] = (
            cleaned_df["due_date"] - cleaned_df["bill_date"]
        ).dt.days
        cleaned_df["gap_eta_due"] = (cleaned_df["due_date"] - cleaned_df["eta"]).dt.days

        if task == "training":
            cleaned_df["shipment_delay_days"] = (
                cleaned_df["receipt_date"] - cleaned_df["eta"]
            ).dt.days
            cleaned_df["shipment_classified"] = cleaned_df["shipment_delay_days"].apply(
                lambda x: "on_time" if x <= self.threshold_for_delay else "delayed"
            )
            cleaned_df.drop(
                columns=date_columns + ["shipment_delay_days", "receipt_date"], inplace=True
            )
        # cleaned_df = fill_unknown_for_object_cols(cleaned_df)
        # print(cleaned_df.info())
        return cleaned_df

    def get_necessary_dataset_for_inference(self) -> pd.DataFrame:
        # query_for_dataset = f"""
        #             SELECT
        #                bni.batch_in_id,
        #                 bni.quantity_in,

        #                 -- From bills
        #                 b.bill_date,
        #                 b.due_date,
        #                 b.eta AS eta,
        #                 b.seal,
        #                 b.customs_broker,
        #                 b.receipt_date,
        #                 b.ocean_freight,

        #                 i.yield_percentage,

        #             FROM
        #                 zoho_books_analytics.batch_number_in bni
        #             JOIN
        #                 zoho_books_analytics.bills b
        #                 ON bni.bill_id = b.bill_id
        #             JOIN
        #                 zoho_books_analytics.bill_item bi
        #                 ON bi.bill_id = b.bill_id
        #             JOIN
        #                 zoho_books_analytics.purchase_orders po
        #                 ON po.purchase_order_number = b.purchase_order
        #             JOIN
        #                 zoho_books_analytics.items i
        #                 ON i.item_id = bi.product_id
        #             JOIN
        #                 zoho_books_analytics.customer_item_mapping cim
        #                 ON i.sku = cim.az_sku

        #             WHERE
        #                 cim.customer_name LIKE '{self.customer_name}'
        #                 AND po.po_commited != '{self.po_comitted}'
        #                 AND bni.created_time BETWEEN '{self.start_timestamp}' AND '{self.end_timestamp}'
        #             ORDER BY bni.created_time DESC
        #     """
        # client = get_clickhouse_client()
        query_for_cubejs = """
                    {{
                    "dimensions": [
                        "DATA_FOR_ML_SERVICES.bill_date",
                        "DATA_FOR_ML_SERVICES.brand",
                        "DATA_FOR_ML_SERVICES.coo",
                        "DATA_FOR_ML_SERVICES.due_date",
                        "DATA_FOR_ML_SERVICES.eta",
                        "DATA_FOR_ML_SERVICES.manufacturer",
                        "DATA_FOR_ML_SERVICES.payment_terms",
                        "DATA_FOR_ML_SERVICES.quantity_in",
                        "DATA_FOR_ML_SERVICES.receipt_date",
                        "DATA_FOR_ML_SERVICES.scac",
                        "DATA_FOR_ML_SERVICES.shipping_port",
                        "DATA_FOR_ML_SERVICES.sku",
                        "DATA_FOR_ML_SERVICES.tariff_amount",
                        "DATA_FOR_ML_SERVICES.shipped_date",
                        "DATA_FOR_ML_SERVICES.batch_in_id",
                        "DATA_FOR_ML_SERVICES.vendor_id",
                        "DATA_FOR_ML_SERVICES.batch_number",
                        "DATA_FOR_ML_SERVICES.purchase_order"
                    ],
                    "filters": [
                        {{
                        "values": ["{customer_name}"],
                        "member": "DATA_FOR_ML_SERVICES.customer_name",
                        "operator": "contains"
                        }}
                    ],
                    "timeDimensions": [
                        {{
                        "dimension": "DATA_FOR_ML_SERVICES.created_time",
                        "dateRange": ["{start_timestamp}", "{end_timestamp}"]
                        }}
                    ]
                    }}
                    """.format(
            customer_name=self.customer_name,
            start_timestamp=self.start_timestamp,
            end_timestamp=self.end_timestamp,
        )
        client = get_cubejs_client()
        try:
            result = client.api_call(
                client.get_base_url() + "?query=" + query_for_cubejs, "GET"
            )
            dataset_df = pd.DataFrame(result.json()["data"])

            # Strip cube name prefix
            dataset_df.columns = [col.split(".")[-1] for col in dataset_df.columns]
            return dataset_df
        except Exception as e:
            print(e)
            return pd.DataFrame()


if __name__ == "__main__":
    dataset = DatasetPreparation(
        start_timestamp="2025-08-01", end_timestamp="2025-08-02"
    )
    cleaned_df = dataset.calculate_target_variable_from_clean_dataset()
    cleaned_df.info()
    # cleaned_df.to_csv("cleaned_dataset.csv", index=False)
