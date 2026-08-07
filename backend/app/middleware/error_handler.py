from fastapi import Request, status
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger("disaster_triage_backend")


async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled Exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred.",
            "error": str(exc),
            "path": request.url.path,
        },
    )
