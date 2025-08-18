from fastapi import APIRouter
from sandhya_aqua.api.recommender import app as recommender_app

router = APIRouter()

router.include_router(
    recommender_app, prefix="/sandhya_aqua", tags=["Sandhya Aqua"]
)   
