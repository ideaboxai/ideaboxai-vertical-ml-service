from fastapi import FastAPI, Request  # Added Request import
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from src.setup_routes import setup_routes
from fastapi.security import HTTPBearer
from dotenv import load_dotenv

load_dotenv()

security = HTTPBearer()


app = FastAPI(
    title="Ideaboxai ML Service API",
    description="API for Ideaboxai ML Services.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    root_path="/",
)

app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=5)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def handle_https_redirects(request: Request, call_next):
    # Check if request is coming through HTTPS reverse proxy
    forwarded_proto = request.headers.get("x-forwarded-proto")
    if forwarded_proto == "https":
        # Override the URL scheme so FastAPI knows it's HTTPS
        request.scope["scheme"] = "https"
        # Also update the server info
        request.scope["server"] = (
            request.headers.get("x-forwarded-host", request.scope["server"][0]),
            443,
        )

    response = await call_next(request)

    # If response is a redirect and we're behind HTTPS proxy, fix the redirect URL
    if (
        response.status_code in [301, 302, 307, 308]
        and forwarded_proto == "https"
        and hasattr(response, "headers")
        and "location" in response.headers
    ):
        location = response.headers["location"]
        if location.startswith("http://"):
            # Replace http:// with https://
            response.headers["location"] = location.replace("http://", "https://", 1)

    return response


setup_routes(app)

if __name__ == "__main__":
    import os
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() == "true"

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        proxy_headers=True,
        forwarded_allow_ips="*",
        log_level=os.getenv("LOG_LEVEL", "info"),
    )
