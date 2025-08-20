import os
import sqlalchemy
from dotenv import load_dotenv
from typing import Literal
from sqlalchemy.ext.asyncio import create_async_engine

load_dotenv()


def get_sandhya_db_engine(connection_mode: Literal["sync", "async"] = "sync"):
    """
    Returns the database engine URL for Sandhya ERP.
    """
    DB_HOST = os.getenv("SANDHYA_ERP_DB_HOST")
    DB_PORT = os.getenv("SANDHYA_ERP_DB_PORT")
    DB_USERNAME = os.getenv("SANDHYA_ERP_DB_USERNAME")
    DB_PASSWORD = os.getenv("SANDHYA_ERP_DB_PASSWORD")
    DB_NAME = os.getenv("SANDHYA_ERP_DB_NAME")

    if not all([DB_HOST, DB_PORT, DB_USERNAME, DB_PASSWORD, DB_NAME]):
        raise ValueError(
            "Sandhya Database connection parameters are not set in the environment variables."
        )

    sandhya_erp_db_url = (
        f"mysql+pymysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    sandhya_erp_db_url_async = (
        f"mysql+aiomysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    if connection_mode == "sync":
        engine = sqlalchemy.create_engine(sandhya_erp_db_url)
    else:
        engine = create_async_engine(sandhya_erp_db_url_async)

    return engine
