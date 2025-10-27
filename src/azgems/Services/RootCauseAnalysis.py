from src.azgems.data_source_connection import get_cubejs_client
from src.azgems.Services.OpenAIclient import OpenAIClient
import json
from pydantic import BaseModel, Field
import logging
import redis

redis_client = redis.Redis(host="ml-service-redis", port=6379, decode_responses=True)
# redis_client = redis.Redis(host="0.0.0.0", port=6379, decode_responses=True)
logger = logging.getLogger(__name__)


class AnalysisResponse(BaseModel):
    issue: str = Field(description="The issue identified in the anomaly.")
    potential_cause: list[str] = Field(
        description="The potential causes of the issue in bullet points."
    )
    recommendation: list[str] = Field(
        description="Recommended actions to resolve the issue in bullet points."
    )


class RootCauseAnalysisService:
    def __init__(self):
        self.cubejs_client = get_cubejs_client()
        self.llm_client = OpenAIClient()

    def get_the_data(self, anomaly_id: str):
        query_str = """
        {{
            "dimensions": [
                "ANOMALIES_DATA.anomaly_root_cause",
                "ANOMALIES_DATA.anomaly_severity",
                "ANOMALIES_DATA.anomaly_type",
                "ANOMALIES_DATA.customer_po",
                "ANOMALIES_DATA.max_wos",
                "ANOMALIES_DATA.message",
                "ANOMALIES_DATA.min_wos",
                "ANOMALIES_DATA.process_stage",
                "ANOMALIES_DATA.selected_customer",
                "ANOMALIES_DATA.sku",
                "ANOMALIES_DATA.source_status",
                "ANOMALIES_DATA.title",
                "ANOMALIES_DATA.vendor_id",
                "ANOMALIES_DATA.wos",
                "ANOMALIES_DATA.year_week_number"
            ],
            "measures": [],
            "filters": [
                {{
                    "member": "ANOMALIES_DATA.id",
                    "operator": "equals",
                    "values": ["{anomaly_id}"]
                }}
            ]
        }}
        """.format(
            anomaly_id=anomaly_id
        )

        result = self.cubejs_client.api_call(
            self.cubejs_client.get_base_url() + "?query=" + query_str, "GET"
        )
        raw_data = result.json().get("data", [])
        if not raw_data:
            return {}

        # --- Clean Keys ---
        cleaned_data = {}
        for key, value in raw_data[0].items():
            new_key = key.replace("ANOMALIES_DATA.", "")
            cleaned_data[new_key] = value

        return cleaned_data

    async def perform_root_cause_analysis(
        self, anomaly_data: dict, anomaly_id: str = ""
    ):
        """
        Perform root cause analysis on anomaly data and return structured JSON response.
        Utilizes Redis caching to avoid redundant LLM calls.
        """
        cache_key = f"rca:{anomaly_id}"

        # --- 1️⃣ Check Redis Cache ---
        cached_result = redis_client.get(cache_key)
        if cached_result:
            logger.info(f"Cache hit for anomaly_id={anomaly_id}")
            return json.loads(cached_result)

        logger.info(f"Cache miss for anomaly_id={anomaly_id}, generating new RCA...")

        # --- 2️⃣ Build Prompts ---
        system_prompt = (
            "You are an expert data analyst specializing in root cause analysis."
        )

        user_prompt = f"""Analyze the following anomaly data and provide insights on 
        the root cause.
        The meaning of WOS is Week of Supply.
        Currently the main focus in on Walmart Inc. as the selected customer.
        The year_week_number represents year and week number in Walmart’s format (e.g. 202546 = 2025 week 46).
        anomaly_root_cause explains why there is an alert.
        inventory status anomaly refers to inventory-related alerts.

        Try to use simple language and avoid jargon.

        Anomaly Data:
        {json.dumps(anomaly_data, indent=2)}

        Your main task is to identify:
        - Issue (one line)
        - Potential causes (bullet points)
        - Recommendations (bullet points)
        """

        # --- 3️⃣ Generate using LLM ---
        try:
            response = await self.llm_client.generate_formatted_response(
                system_prompt, user_prompt, AnalysisResponse
            )

            response_json = {
                "issue": getattr(response, "issue", None),
                "potential_causes": getattr(response, "potential_cause", None),
                "recommended_actions": getattr(response, "recommendation", None),
            }

            # --- 4️⃣ Store in Redis ---
            redis_client.setex(
                cache_key, 86400, json.dumps(response_json)
            )  # expires in 1 day
            logger.info(f"Stored RCA in cache for anomaly_id={anomaly_id}")

            return response_json

        except Exception as e:
            logger.exception("Error performing root cause analysis")
            return {
                "error": str(e),
                "issue": None,
                "potential_causes": None,
                "recommended_actions": None,
            }


if __name__ == "__main__":
    import asyncio

    rca_service = RootCauseAnalysisService()
    anomaly_data = rca_service.get_the_data("0369a871-63f8-4816-b2a9-6b1d639f4ea3")
    data = asyncio.run(
        rca_service.perform_root_cause_analysis(
            anomaly_data, "0369a871-63f8-4816-b2a9-6b1d639f4ea3"
        )
    )
    print(json.dumps(data, indent=2))
