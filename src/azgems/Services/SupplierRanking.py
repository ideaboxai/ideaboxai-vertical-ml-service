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
        supplier_data_query = f"""
        SELECT
            v.vendor_name,
            
            -- Total quantity received
            SUM(CAST(ifNull(bni.quantity_in, 0) AS Float64)) AS total_quantity_in,

            -- Avg date differences
            ROUND(
                AVG(
                    dateDiff(
                        'day',
                        parseDateTimeBestEffortOrNull(b.shipped_date),
                        parseDateTimeBestEffortOrNull(b.receipt_date)
                    )
                ), 2
            ) AS avg_shipped_receipt_diff_days,
            
            ROUND(
                AVG(
                    dateDiff(
                        'day',
                        parseDateTimeBestEffortOrNull(b.shipped_date),
                        parseDateTimeBestEffortOrNull(b.bill_date)
                    )
                ), 2
            ) AS avg_shipped_bill_diff_days,

            -- Vendor relationship duration
            dateDiff('month', MIN(b.created_time), today()) AS using_goods_since_months,

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
            ) AS on_time_delivery_rate_percent,

            -- Distinct products
            COUNT(DISTINCT bi.product_id) AS distinct_items_supplied,
            COUNT(DISTINCT b.bill_id) AS total_shipments
        FROM 
            zoho_books_analytics.batch_number_in AS bni
        JOIN 
            zoho_books_analytics.bills AS b 
            ON bni.bill_id = b.bill_id
        JOIN 
            zoho_books_analytics.bill_item AS bi 
            ON bi.bill_id = b.bill_id
        JOIN 
            zoho_books_analytics.purchase_orders AS po  
            ON po.purchase_order_number = b.purchase_order
        JOIN 
            zoho_books_analytics.items AS i 
            ON i.item_id = bi.product_id
        JOIN 
            zoho_books_analytics.customer_item_mapping AS cim 
            ON i.sku = cim.az_sku
        JOIN 
            zoho_books_analytics.vendors AS v 
            ON v.vendor_id = b.vendor_id 
        WHERE 
            cim.customer_name LIKE '{self.customer_name}%'
            AND po.po_commited != '{self.po_comitted}'
            AND b.receipt_date != '' 
            AND b.eta != ''
        GROUP BY 
            v.vendor_name
        ORDER BY 
            total_quantity_in DESC
        """
        result = self.clickhouse_client.query(query=supplier_data_query)

        return pd.DataFrame(
            result.result_rows, columns=[col for col in result.column_names]
        )

    def get_supplier_ranking_cubejs(self):
        supplier_data_query = """
        SELECT * FROM supplier_ranking_cubejs
        """
        return self.cubejs_client.query(supplier_data_query).result_rows

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
        supplier_data_df = self.get_supplier_ranking_data()
        numeric_cols = supplier_data_df.select_dtypes(include=["number"]).columns
        supplier_data_df[numeric_cols] = supplier_data_df[numeric_cols].fillna(
            supplier_data_df[numeric_cols].median()
        )  # one supplier has null date fields
        X = supplier_data_df.iloc[:, 1:].values
        weights = [0.13, 0.13, 0.08, 0.09, 0.13, 0.22, 0.09, 0.13]
        types = [1, -1, -1, 1, 1, 1, 1, 1]
        X = supplier_data_df.iloc[:, 1:].values
        rank = self.topsis(X, weights=weights, types=types)
        supplier_data_df["topsis_score"] = np.round(rank * 100, 2)
        sorted_df = supplier_data_df.sort_values(by="topsis_score", ascending=False)
        return sorted_df.to_dict(orient="records")


if __name__ == "__main__":
    supplier_ranking = SupplierRanking()
    print(supplier_ranking.calculate_topsis_score())
