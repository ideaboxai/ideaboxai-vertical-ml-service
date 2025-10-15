from src.azgems.data_source_connection import get_clickhouse_client, get_cubejs_client
import pandas as pd
from typing import Literal


def fill_unknown_for_object_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].replace(r"^\s*$", "unknown", regex=True)
    return df


class DatasetPreparation:
    def __init__(
        self,
        customer_name="Walmart",
        po_comitted="Direct Sale",
        threshold=5,
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
                        "DATA_FOR_ML_SERVICES.seal",
                        "DATA_FOR_ML_SERVICES.eta",
                        "DATA_FOR_ML_SERVICES.batch_in_id",
                        "DATA_FOR_ML_SERVICES.bill_date",
                        "DATA_FOR_ML_SERVICES.created_time",
                        "DATA_FOR_ML_SERVICES.customer_name",
                        "DATA_FOR_ML_SERVICES.customs_broker",
                        "DATA_FOR_ML_SERVICES.due_date",
                        "DATA_FOR_ML_SERVICES.ocean_freight",
                        "DATA_FOR_ML_SERVICES.po_commited",
                        "DATA_FOR_ML_SERVICES.quantity_in",
                        "DATA_FOR_ML_SERVICES.receipt_date",
                        "DATA_FOR_ML_SERVICES.yield_percentage"
                    ],
                    "filters": [
                        {{
                            "values": ["{customer_name}"],
                            "member": "DATA_FOR_ML_SERVICES.customer_name",
                            "operator": "contains"
                        }},
                        {{
                            "values": ["{po_comitted}"],
                            "member": "DATA_FOR_ML_SERVICES.po_commited",
                            "operator": "notContains"
                        }}
                    ],
                    "measures": []
                }}""".format(
            customer_name=self.customer_name, po_comitted=self.po_comitted
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
        dataset["yield_percentage"] = (
            dataset["yield_percentage"]
            .astype(str)
            .str.strip()
            .replace("", "0")
            .str.replace("%", "", regex=False)
            .fillna("0")
            .astype(float)
        )
        dataset["quantity_in"] = dataset["quantity_in"].astype(float)
        dataset["ocean_freight"] = (
            dataset["ocean_freight"]
            .astype(str)
            .str.strip()
            .replace("", "0")
            .fillna("0")
            .astype(float)
        )
        return dataset

    def calculate_target_variable_from_clean_dataset(
        self, task: Literal["training", "inference"] = "training"
    ):
        cleaned_df = self.clean_dataset(task=task)
        cleaned_df = cleaned_df.dropna(subset=["receipt_date", "eta"])
        # converting date columns to datetime
        date_columns = ["bill_date", "due_date", "eta"]
        for col in date_columns:
            cleaned_df[col] = pd.to_datetime(
                cleaned_df[col], format="%d %b %Y", errors="coerce"
            )
            cleaned_df[f"{col}_year"] = cleaned_df[f"{col}"].dt.year
            cleaned_df[f"{col}_year"] = cleaned_df[f"{col}_year"].astype("str")
            cleaned_df[f"{col}_month"] = cleaned_df[f"{col}"].dt.month
            cleaned_df[f"{col}_month"] = cleaned_df[f"{col}_month"].astype("str")
            cleaned_df[f"{col}_day"] = cleaned_df[col].dt.day
            cleaned_df[f"{col}_day"] = cleaned_df[f"{col}_day"].astype("str")
            cleaned_df[f"{col}_weekday"] = cleaned_df[f"{col}"].dt.weekday
            cleaned_df[f"{col}_weekday"] = cleaned_df[f"{col}_weekday"].astype("str")

        cleaned_df["receipt_date"] = pd.to_datetime(
            cleaned_df["receipt_date"], format="%d %b %Y", errors="coerce"
        )
        cleaned_df["shipment_delay_days"] = (
            cleaned_df["receipt_date"] - cleaned_df["eta"]
        ).dt.days
        cleaned_df["shipment_classified"] = cleaned_df["shipment_delay_days"].apply(
            lambda x: "on_time" if x <= self.threshold_for_delay else "delayed"
        )
        cleaned_df.drop(
            columns=date_columns + ["shipment_delay_days", "receipt_date"], inplace=True
        )
        cleaned_df = fill_unknown_for_object_cols(cleaned_df)
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
        query_for_cubejs = """{{
                        "dimensions": [
                "DATA_FOR_ML_SERVICES.seal",
                "DATA_FOR_ML_SERVICES.eta",
                "DATA_FOR_ML_SERVICES.batch_in_id",
                "DATA_FOR_ML_SERVICES.bill_date",
                "DATA_FOR_ML_SERVICES.created_time",
                "DATA_FOR_ML_SERVICES.customer_name",
                "DATA_FOR_ML_SERVICES.customs_broker",
                "DATA_FOR_ML_SERVICES.due_date",
                "DATA_FOR_ML_SERVICES.ocean_freight",
                "DATA_FOR_ML_SERVICES.po_commited",
                "DATA_FOR_ML_SERVICES.quantity_in",
                "DATA_FOR_ML_SERVICES.receipt_date",
                "DATA_FOR_ML_SERVICES.yield_percentage",
                "DATA_FOR_ML_SERVICES.vendor_id",
                "DATA_FOR_ML_SERVICES.batch_number",
                "DATA_FOR_ML_SERVICES.sku",
                "DATA_FOR_ML_SERVICES.purchase_order"
            ],
            "filters": [
                {{
                    "values": ["{customer_name}"],
                    "member": "DATA_FOR_ML_SERVICES.customer_name",
                    "operator": "contains"
                }},
                {{
                    "values": ["{po_comitted}"],
                    "member": "DATA_FOR_ML_SERVICES.po_commited",
                    "operator": "notContains"
                }},
                {{
                    "values": ["{start_timestamp}", "{end_timestamp}"],
                    "member": "DATA_FOR_ML_SERVICES.created_time",
                    "operator": "inDateRange"
                }}
            ],
            "measures": []
        }}""".format(
            customer_name=self.customer_name,
            po_comitted=self.po_comitted,
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
