import os
import requests
from dotenv import load_dotenv

load_dotenv()


class CubeJSClient:
    def __init__(self, token: str = None, base_url: str = None):
        self.base_url = base_url or os.getenv("CUBE_BASE_URL")
        self.access_token = token or os.getenv("CUBE_TOKEN")

    def get_base_url(self) -> str:
        return self.base_url

    def api_call(self, url, api_request_method, query_params=None, request_body=None):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token}",
        }
        try:
            if api_request_method == "GET":
                response = requests.get(url, headers=headers, params=query_params)

            elif api_request_method == "DELETE":
                response = requests.delete(url, headers=headers, params=query_params)

            elif api_request_method == "PUT":
                response = requests.put(url, headers=headers, json=request_body)

            else:
                response = requests.post(
                    url, headers=headers, json=request_body, params=query_params
                )

        except Exception as ex:
            raise ValueError(f"Error on calling API {url}:- {ex}")

        return response
