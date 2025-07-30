from fastapi import FastAPI
from contextlib import asynccontextmanager
from api.v1.endpoints import stout
from stout_service import stout_service

@asynccontextmanager
async def lifespan(app):
    print("Starting up STOUT API...")
    # Load models on startup to avoid cold starts
    stout_service.load_models()
    print("STOUT API ready!")
    yield

app = FastAPI(
    title="STOUT API",
    description="API for STOUT (SMILES to IUPAC and IUPAC to SMILES) translation",
    version="1.0.0",
    lifespan=lifespan
)

# Include routers
app.include_router(stout.router)

@app.get("/")
def root():
    return {
        "message": "STOUT API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/stout/health"
    } 