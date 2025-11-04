import os
import requests
from dotenv import load_dotenv
import time

load_dotenv()


class CubeJSClient:
    def __init__(self, token: str = None, base_url: str = None):
        self.base_url = base_url or os.getenv("CUBEJS_BASE_URL")
        self.access_token = token or os.getenv("CUBEJS_TOKEN")

    def get_base_url(self) -> str:
        return self.base_url

    def api_call(self, url, api_request_method, payload_data=None, query_params=None):
        """Make API call to the serverless endpoint."""
        # Can implement exponential backoff retry logic,..... here
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token}",
        }
        try:
            if api_request_method == "GET":
                # self.logger(f"Calling GET API: {url} with payload: {payload_data}")

                status = True
                while status:
                    response = requests.get(url, headers=headers, params=payload_data)
                    if response.status_code == 200 and "error" in response.json():
                        print(
                            "Cubejs response with continue wait. retrying after a seconds"
                        )
                        time.sleep(2)

                    else:
                        status = False
                return response

            elif api_request_method == "DELETE":
                # self.logger.info(f"Calling DELETE API: {url} with payload: {payload_data}")
                response = requests.delete(url, headers=headers, params=payload_data)

            elif api_request_method == "PUT":
                response = requests.put(url, headers=headers, json=payload_data)

            else:
                # self.logger.info(f"Calling POST API: {url} with payload: {payload_data}")
                response = requests.post(
                    url, headers=headers, json=payload_data, params=query_params
                )

        except Exception as ex:
            raise ValueError(f"Error on calling API {url}:- {ex}")

        return response
