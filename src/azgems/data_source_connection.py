import os
import clickhouse_connect
from dotenv import load_dotenv
from src.shared.clients.cubejs_client import CubeJSClient

load_dotenv()


def get_clickhouse_client():
    """
    Returns a ClickHouse client using environment variables.
    """
    # Load connection parameters once
    config = {
        "host": os.getenv("CLICKHOUSE_HOST"),
        "port": int(os.getenv("CLICKHOUSE_PORT", "8123")),
        "username": os.getenv("CLICKHOUSE_USERNAME"),
        "password": os.getenv("CLICKHOUSE_PASSWORD"),
        "database": os.getenv("CLICKHOUSE_DATABASE"),
    }

    if not all(config.values()):
        missing = [k for k, v in config.items() if not v]
        raise ValueError(f"Missing ClickHouse config(s): {', '.join(missing)}")

    return clickhouse_connect.get_client(**config)


def get_cubejs_client():
    return CubeJSClient(
        token=os.getenv("CUBEJS_TOKEN"),
        base_url=os.getenv("AZGEMS_CUBEJS_BASE_URL"),
    )


if __name__ == "__main__":
    # client = get_clickhouse_client()
    # print("Connected:", client.query("SELECT 1").result_rows)
    client = get_cubejs_client()
    print(
        "Connected:",
        client.api_call(
            "https://cube-dev-demand-forecast.ideaboxai.com/cubejs-api/v1/load", "GET"
        ),
    )
