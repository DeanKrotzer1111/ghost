"""Ghost Trading System — FastAPI entry point."""
import uvicorn
from ghost.config.settings import settings


def main():
    uvicorn.run(
        "ghost.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
