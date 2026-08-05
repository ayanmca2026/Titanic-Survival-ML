import time
import uuid
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

# Setup logging (safe for Vercel / serverless)
try:
    log_dir = Path("/tmp/logs" if os.name != "nt" else "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    api_log_path = log_dir / "api.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(api_log_path),
            logging.StreamHandler()
        ]
    )
except Exception:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )

logger = logging.getLogger("api")

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        logger.info(f"Request started: {request_id} - {request.method} {request.url.path}")
        
        try:
            response = await call_next(request)
            
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Request-ID"] = request_id
            
            logger.info(
                f"Request completed: {request_id} - {request.method} {request.url.path} "
                f"- Status: {response.status_code} - Latency: {process_time:.4f}s"
            )
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Request failed: {request_id} - {request.method} {request.url.path} "
                f"- Error: {str(e)} - Latency: {process_time:.4f}s"
            )
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error", "request_id": request_id}
            )

def setup_middleware(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestLoggingMiddleware)
