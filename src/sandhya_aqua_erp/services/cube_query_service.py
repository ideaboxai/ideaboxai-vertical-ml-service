import os
import json
import urllib.parse
import httpx
from dotenv import load_dotenv
from src.sandhya_aqua_erp.utils.cube_queries import LLMRecommendationCubeQueries as CubeQueries

load_dotenv()


class CubeService:
    def __init__(self, base_url: str = None, token: str = None):
        self.base_url = base_url or os.getenv("CUBE_BASE_URL")
        self.token = token or os.getenv("CUBE_TOKEN")

        if not self.base_url:
            raise EnvironmentError("CUBE_BASE_URL is missing in environment variables")
        if not self.token:
            raise EnvironmentError("CUBE_TOKEN is missing in environment variables")

    @staticmethod
    def _clean_data(data_list: list) -> list:
        """Remove 'RECOMMENDATION.' prefix from keys"""
        return [
            {key.replace("RECOMMENDATION.", ""): value for key, value in item.items()}
            for item in data_list
        ]

    async def get_data(self, query_key: str, lot_number: str = None, lot_filter:json=None) -> list:
        """Fetch data from Cube service for given query and lot_number"""
        try:
            if query_key not in CubeQueries.keys():
                raise KeyError(f"Invalid query_key: {query_key}")

            query_dict = json.loads(CubeQueries[query_key])
            
            if lot_filter:
                query_dict["filters"].append(lot_filter)
                
            query_str = json.dumps(query_dict)
            encoded_query = urllib.parse.quote(query_str)
            url = f"{self.base_url}{encoded_query}"

            headers = {"Authorization": self.token, "Content-Type": "application/json"}

            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()

            if "data" not in data:
                raise KeyError(
                    f"Expected 'data' in response, got keys: {list(data.keys())}"
                )

            return self._clean_data(data["data"])

        except Exception as e:
            print(f"Error fetching data for {query_key}: {e}")
            return None
