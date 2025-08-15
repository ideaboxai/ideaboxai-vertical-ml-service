TABLES_FEATURES_STRATEGY_MAPPER = {
    "erpx_dev_rm_procurement.grn": {"features_to_look": [], "strategy": None},
    "erpx_dev_rm_procurement.grn_items": {
        "features_to_look": [
            "quantity",
            "count",
            "received_boxes",
            "soft_percentage",
            "boxes",
            "antibiotic_test",
            "fungus_percentage",
        ],
        "strategy": ["univariate", "multivariate"],
    },
    "erpx_dev_rm_procurement.indent": {"features_to_look": [], "strategy": None},
    "erpx_dev_rm_procurement.indent_items": {
        "features_to_look": ["expected_count", "expected_qty", "expected_price"],
        "strategy": ["univariate", "multivariate"],
    },
    # ----------------- Inward related tables -----------------
    "erpx_dev_production.pp_inward": {
        "features_to_look": [
            "weight_from_gate",
            "count_at_farm_gate",
            "trays_received_from_farm",
            "verified_weight",
            "verified_net_weight",
            "verified_count",
            "verified_no_of_trays",
            "sample_count",
        ],
        "strategy": ["univariate", "multivariate"],
    },
    "erpx_dev_production.pp_inward_reading": {
        "features_to_look": ["weight", "total_weight", "crates"],
        "strategy": ["univariate", "multivariate"],
    },
    # ----------------- Grading related tables -----------------
    "erpx_dev_production.pp_grading": {
        "features_to_look": ["crate_weight"],
        "strategy": ["univariate"],
    },
    "erpx_dev_production.pp_grading_grades": {
        "features_to_look": ["count", "uniformity_ratio"],
        "strategy": ["univariate", "multivariate"],
    },
    "erpx_dev_production.pp_grading_readings": {
        "features_to_look": ["count", "weight", "crates"],
        "strategy": ["univariate", "multivariate"],
    },
    # ----------------- Soaking related tables -----------------
    "erpx_dev_production.pp_soaking_lot": {
        "features_to_look": ["count"],
        "strategy": ["univariate"],
    },
    "erpx_dev_production.pp_soaking": {
        "features_to_look": ["soak_time", "crate_weight", "soakin_count"],
        "strategy": ["univariate", "multivariate"],
    },
    "erpx_dev_production.pp_soaking_readings": {
        "features_to_look": ["weight", "crates"],
        "strategy": ["univariate", "multivariate"],
    },
    "erpx_dev_production.pp_soaking_temperature": {
        "features_to_look": ["temperature"],
        "strategy": ["univariate"],
    },
    # ----------------- Cooking related tables -----------------
    "erpx_dev_production.pp_cooking_lot": {
        "features_to_look": ["count"],
        "strategy": ["univariate"],
    },
    "erpx_dev_production.pp_cooking": {
        "features_to_look": ["cooking_temp", "chilling_temp"],
        "strategy": ["univariate", "multivariate"],
    },
    "erpx_dev_production.pp_cooking_readings": {
        "features_to_look": ["temperature", "temperature_type", "temperature_fault"],
        "strategy": ["multivariate"],
    },
    # ----------------- Packing related tables -----------------
    "erpx_dev_production.pp_packing": {
        "features_to_look": [
            "net_weight",
            "min_weight",
            "no_of_pouches",
            "pouch_weight",
            "min_glaze",
            "max_glaze",
        ],
        "strategy": ["univariate", "multivariate"],
    },
    # 'erpx_dev_production.pp_packing_reading': ['weight', 'crates'],
    "erpx_dev_production.pp_packing_master_readings": {
        "features_to_look": ["weight", "weight_limit"],
        "strategy": ["univariate", "multivariate"],
    },
    "erpx_dev_production.pp_packing_glazing_reading": {
        "features_to_look": [
            "frozen_weight",
            "de_glazed_weight",
            "glaze_percentage",
            "duration",
        ],
        "strategy": ["univariate", "multivariate"],
    },
    "erpx_dev_production.pp_packing_metal_detector": {
        "features_to_look": [],
        "strategy": None,
    },
}
