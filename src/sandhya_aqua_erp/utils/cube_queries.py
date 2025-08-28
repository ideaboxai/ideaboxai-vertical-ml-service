LLMRecommendationCubeQueries = {
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
   
  ]
}
""",
"anomaly_query":"""{
  "dimensions": [
    "ANOMALY_NUMBER.title",
    "ANOMALY_NUMBER.status",
    "ANOMALY_NUMBER.anomaly_type",
    "ANOMALY_NUMBER.anomaly_root_cause",
    "ANOMALY_NUMBER.lot_number",
    "ANOMALY_NUMBER.customer_po",
    "ANOMALY_NUMBER.anomaly_severity",
    "ANOMALY_NUMBER.process_stage",
    "ANOMALY_NUMBER.message",
    "ANOMALY_NUMBER.sku"
  ],
  "order": {
    "ANOMALY_NUMBER.total_records": "desc"
  },
  "filters": [
    {
      "member": "ANOMALY_NUMBER.status",
      "operator": "equals",
      "values": [
        "ACTIVE"
      ]
    }
  ]
}"""
}