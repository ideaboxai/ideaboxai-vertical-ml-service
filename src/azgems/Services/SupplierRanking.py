from src.azgems.data_source_connection import get_clickhouse_client, get_cubejs_client
import pandas as pd
from pymcdm.methods import TOPSIS
import numpy as np


class SupplierRanking:
    def __init__(self, customer_name="Walmart", po_comitted="Direct Sale"):
        self.clickhouse_client = get_clickhouse_client()
        self.cubejs_client = get_cubejs_client()
        self.topsis = TOPSIS()
        self.customer_name = customer_name
        self.po_comitted = po_comitted

    def get_supplier_ranking_data(self):
        supplier_ranking_query = f"""SELECT
                v.vendor_name,
                SUM(CAST(ifNull(bni.quantity_in, 0) AS Float64)) AS total_quantity_received,
                ROUND(
                    AVG(
                        dateDiff(
                            'day',
                            parseDateTimeBestEffortOrNull(b.shipped_date),
                            parseDateTimeBestEffortOrNull(b.receipt_date)
                        )
                    ), 2
                ) AS avg_days_for_shipment_to_arrive,
                -- Net monetary transaction
                ROUND(
                    SUM(
                        CAST(
                            replaceAll(replaceAll(b.total_bcy, 'USD ', ''), ',', '') AS Float64
                        )
                        -
                        CAST(
                            replaceAll(replaceAll(ifNull(b.discount_amount_bcy, '0'), 'USD ', ''), ',', '') AS Float64
                        )
                    ), 2
                ) AS net_monetary_transaction,
                -- On-time delivery rate
                ROUND(
                    (countIf(
                        parseDateTimeBestEffortOrNull(b.receipt_date) <= parseDateTimeBestEffortOrNull(b.eta)
                    ) * 100.0)
                    / nullIf(countIf(b.receipt_date != '' AND b.eta != ''), 0),
                    2
                ) AS on_time_delivery_rate,
                -- Distinct products
                COUNT(DISTINCT bi.product_id) AS distinct_items_supplied,
            COUNT(DISTINCT b.bill_id) AS total_shipments,
            ROUND(
                    COUNT(DISTINCT b.bill_id) 
                    / nullIf(dateDiff('year', MIN(b.created_time), today()), 0),
                    2
                ) AS avg_shipments_per_year
            FROM zoho_books_analytics.batch_number_in bni
            INNER JOIN zoho_books_analytics.bills b on bni.bill_id = b.bill_id 
            INNER JOIN zoho_books_analytics.bill_item bi ON b.bill_id  = bi.bill_id
            INNER JOIN zoho_books_analytics.purchase_orders p ON b.purchase_order   = p.purchase_order_number  
            INNER JOIN zoho_books_analytics.items i ON bi.product_id   = i.item_id 
            INNER JOIN zoho_books_analytics.sales_orders so ON p.reference_number = so.sales_order 
            INNER JOIN zoho_books_analytics.customers c ON c.customer_id   = so.customer_id
            INNER JOIN zoho_books_analytics.customer_item_mapping ci ON i.sku   = ci.az_sku 
            INNER JOIN zoho_books_analytics.vendors v on v.vendor_id = b.vendor_id
            WHERE	 c.customer_name   like '{self.customer_name}%'
                AND b.receipt_date != '' 
                AND b.eta != ''
            GROUP BY 
                v.vendor_name
            ORDER BY 
                total_quantity_received DESC"""
        result = self.clickhouse_client.query(query=supplier_ranking_query)

        return pd.DataFrame(
            result.result_rows, columns=[col for col in result.column_names]
        )

    def get_supplier_ranking_cubejs(self):
        supplier_data_query = """
                {{
                    "dimensions": [
                        "SUPPLIER_RANKING_DATA.vendor_name",
                        "SUPPLIER_RANKING_DATA.avg_days_for_shipment_to_arrive",
                        "SUPPLIER_RANKING_DATA.net_monetary_transaction",
                        "SUPPLIER_RANKING_DATA.on_time_delivery_rate",
                        "SUPPLIER_RANKING_DATA.distinct_items_supplied",
                        "SUPPLIER_RANKING_DATA.total_shipments",
                        "SUPPLIER_RANKING_DATA.avg_shipments_per_year"
                    ],
                    "measures": [
                        "SUPPLIER_RANKING_DATA.total_quantity_received"
                    ],
                    "filters": [
                        {{
                            "member": "SUPPLIER_RANKING_DATA.customer_name",
                            "operator": "contains",
                            "values": ["{customer_name}"]
                        }}
                    ]
                }}
                """.format(
            customer_name=self.customer_name
        )
        try:
            result = self.cubejs_client.api_call(
                self.cubejs_client.get_base_url() + "?query=" + supplier_data_query,
                "GET",
            )
            dataset_df = pd.DataFrame(result.json()["data"])

            dataset_df.columns = [col.split(".")[-1] for col in dataset_df.columns]
            return dataset_df
        except Exception as e:
            print(e)
            raise Exception(f"Error getting necessary dataset for training: {e}")

    def calculate_topsis_score(self):
        # weights = {
        # "total_shipments": 0.13
        # "distinct_items_supplied": 0.09
        # "on_time_delivery_rate_percent": 0.22
        # "net_monetary_transaction": 0.13
        # "using_goods_since_months": 0.09
        # "avg_shipped_bill_diff_days": 0.08
        # "avg_shipped_receipt_diff_days": 0.13
        # "total_quantity_in": 0.13
        # }

        supplier_data_df = self.get_supplier_ranking_cubejs()

        keep_columns = [
            "vendor_name",
            "total_quantity_received",
            "avg_days_for_shipment_to_arrive",
            "net_monetary_transaction",
            "on_time_delivery_rate",
            "distinct_items_supplied",
            "avg_shipments_per_year",
            "total_shipments",
        ]
        supplier_data_df = supplier_data_df[keep_columns]
        weights = [0.15, 0.15, 0.15, 0.22, 0.1, 0.13, 0.1]
        types = [1, -1, 1, 1, 1, 1, 1]

        X = supplier_data_df.iloc[:, 1:].values

        rank = self.topsis(X, weights=weights, types=types)
        supplier_data_df["topsis_score"] = np.round(rank * 100, 2)

        sorted_df = supplier_data_df.sort_values(by="topsis_score", ascending=False)

        return sorted_df.to_dict(orient="records")


if __name__ == "__main__":
    supplier_ranking = SupplierRanking()
    print(supplier_ranking.calculate_topsis_score())
