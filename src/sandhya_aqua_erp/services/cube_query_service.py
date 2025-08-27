import requests
import json
import urllib.parse
import os


cube_queries = {
    "grn_process_query": """{
  "dimensions": [
    "RECOMMENDATION.plant_lot_number",
    "RECOMMENDATION.quantity_in_kg_at_GRN",
    "RECOMMENDATION.number_of_shrimps_in_a_kg_at_GRN",
    "RECOMMENDATION.vi_inspection_during_GRN",
    "RECOMMENDATION.rm_inspection_during_GRN",
    "RECOMMENDATION.antibiotic_test_during_GRN",
    "RECOMMENDATION.loose_shell_during_GRN",
    "RECOMMENDATION.fungus_percentage_during_GRN",
    "RECOMMENDATION.sale_order"
  ],
  "order": {
    "RECOMMENDATION.plant_lot_number": "asc"
  },
  "filters": [
    {
      "member": "RECOMMENDATION.sale_order",
      "operator": "contains",
      "values": []
    }
  ]
}""",
    "grading_process_query": """{
  "dimensions": [
    "RECOMMENDATION.plant_lot_number",
    "RECOMMENDATION.weight_of_each_crate",
    "RECOMMENDATION.weight_of_shrimp_in_kg_at_grading",
    "RECOMMENDATION.data_at_shrimp_grading_process"
  ],
  "order": {
    "RECOMMENDATION.plant_lot_number": "asc"
  },
  "filters": [
    {
      "member": "RECOMMENDATION.sale_order",
      "operator": "contains",
      "values": []
    }
  ]
}""",
    "soaking_process_query": """{
  "dimensions": [
    "RECOMMENDATION.plant_lot_number",
    "RECOMMENDATION.threshold_time_in_minutes_for_soaking_shrimps",
    "RECOMMENDATION.time_in_minutes_for_soaking_shrimps",
    "RECOMMENDATION.tub_in_which_the_lot_was_soaked",
    "RECOMMENDATION.ingridients_of_solution",
    "RECOMMENDATION.purchase_order_of_customer",
    "RECOMMENDATION.stock_keeping_unit",
    "RECOMMENDATION.soaking_weight",
    "RECOMMENDATION.soaking_process_type",
    "RECOMMENDATION.soaking_status"
  ],
  "order": {
    "RECOMMENDATION.lot_number": "asc"
  },
  "filters": [
    {
      "member": "RECOMMENDATION.sale_order",
      "operator": "contains",
      "values": []
    }
  ]
}""",
    "cooking_process_query": """
    {
  "dimensions": [
    "RECOMMENDATION.plant_lot_number",
    "RECOMMENDATION.status_of_cooking",
    "RECOMMENDATION.start_of_cooking",
    "RECOMMENDATION.end_of_cooking",
    "RECOMMENDATION.cooking_created_at",
    "RECOMMENDATION.temperature_that_should_be_at_the_time_of_cooling_shrimps",
    "RECOMMENDATION.temperature_that_should_be_at_the_time_of_cooking_shrimps",
    "RECOMMENDATION.last_5_temperature_readings_in_different_parts_of_cooking_process"
  ],
  "order": {
    "RECOMMENDATION.plant_lot_number": "asc"
  },
  "filters": [
    {
      "member": "RECOMMENDATION.sale_order",
      "operator": "contains",
      "values": []
    }
  ]
}""",
    "yield_calculation_query": """{
  "dimensions": [
    "RECOMMENDATION.plant_lot_number",
    "RECOMMENDATION.grn_created_dt",
    "RECOMMENDATION.hon_count",
    "RECOMMENDATION.hon_weight",
    "RECOMMENDATION.hl_weight",
    "RECOMMENDATION.grading_yield",
    "RECOMMENDATION.soaking_cnt",
    "RECOMMENDATION.soaking_weight",
    "RECOMMENDATION.grading_count"
  ],
  "order": {
    "RECOMMENDATION.grn_created_dt": "asc",
    "RECOMMENDATION.plant_lot_number": "asc"
  },
  "filters": [
    {
      "member": "RECOMMENDATION.sale_order",
      "operator": "contains",
      "values": []
    }
  ]
}
""",
}
import os
import json
import urllib.parse
import requests
from dotenv import load_dotenv
import httpx

load_dotenv()


def clean_data_from_cube_query(data_list):
    cleaned_data = []
    for item in data_list:
        cleaned_item = {}
        for key, value in item.items():
            new_key = key.replace("RECOMMENDATION.", "")
            cleaned_item[new_key] = value
        cleaned_data.append(cleaned_item)
    return cleaned_data

async def get_data_from_cube_query(query_key, lot_number, sale_order):
    try:
        if query_key not in cube_queries:
            raise KeyError(f"Invalid query_key: {query_key}")

        query_dict = json.loads(cube_queries[query_key])
        query_dict["filters"][0]["values"] = [sale_order]

        if lot_number:
            lot_filter = {
                "member": "RECOMMENDATION.plant_lot_number",
                "operator": "equals",
                "values": [lot_number],
            }
            query_dict["filters"].append(lot_filter)

        query_str = json.dumps(query_dict)

        base_url = os.getenv("CUBE_BASE_URL")
        encoded_query = urllib.parse.quote(query_str)
        url = f"{base_url}{encoded_query}"

        token = os.getenv("CUBE_TOKEN")
        if not token:
            raise EnvironmentError("CUBE_TOKEN is missing in environment variables")

        headers = {"Authorization": token, "Content-Type": "application/json"}

        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

        if "data" not in data:
            raise KeyError(f"Expected 'data' in response, got keys: {list(data.keys())}")

        return clean_data_from_cube_query(data["data"])

    except Exception as e:
        print(f"Error in get_data_from_cube_query: {e}")
        return None
