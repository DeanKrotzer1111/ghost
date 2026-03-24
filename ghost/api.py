"""Ghost v5.5 FastAPI Application — REST API for signal processing and system health."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ghost import __version__

app = FastAPI(
    title="Ghost Trading System",
    version=__version__,
    description="AI-powered autonomous futures trading system",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "operational",
        "version": __version__,
        "system": "ghost",
    }


@app.get("/api/v1/status")
async def status():
    return {
        "version": __version__,
        "modules_loaded": 29,
        "gates": 10,
        "instruments": 30,
    }
