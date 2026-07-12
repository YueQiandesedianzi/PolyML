"""PolyML Backend — FastAPI Application Entry Point"""

import os
# Prevent BLAS/threading crash on Windows conda — must be before numpy import
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import settings
from db.database import init_db, close_db
from routers.project import router as project_router
from routers.data import router as data_router
from routers.features import router as features_router
from routers.automl import router as automl_router
from routers.prediction import router as prediction_router
from routers.model_io import router as model_io_router
from routers.polymer_db import router as polymer_db_router
from routers.doe import router as doe_router
from routers.custom_features import router as custom_features_router
from routers.active_learning import router as active_learning_router
from routers.code_export import router as code_export_router
from routers.agent import router as agent_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db(settings.app_data_path)
    print(f"[PolyML] Database initialized at {settings.app_data_path}")

    # Seed built-in polymer data
    from db.seed_polymers import seed_polymers
    from db.database import async_session
    async with async_session() as session:
        await seed_polymers(session)

    yield
    # Shutdown
    await close_db()


app = FastAPI(
    title="PolyML",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:18921", "http://127.0.0.1:18921"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(project_router)
app.include_router(data_router)
app.include_router(features_router)
app.include_router(automl_router)
app.include_router(prediction_router)
app.include_router(model_io_router)
app.include_router(polymer_db_router)
app.include_router(doe_router)
app.include_router(custom_features_router)
app.include_router(active_learning_router)
app.include_router(code_export_router)
app.include_router(agent_router)


@app.get("/api/health")
async def health_check():
    rdkit_ok = False
    try:
        from rdkit import Chem
        rdkit_ok = Chem.MolFromSmiles("CCO") is not None
    except Exception:
        pass

    return {
        "status": "ok",
        "rdkit": rdkit_ok,
        "version": "0.1.0",
    }
