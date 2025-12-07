from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.core.database import init_db, AsyncSessionLocal
from app.core.seed import seed_data
from app.ml.loader import MLLoader
from app.api.router import api_router

tags_metadata = [
    {
        "name": "Predictive Analytics",
        "description": "ML-ядро: Прогнозы, Антифрод, Классификация.",
    },
    {
        "name": "Operational Data",
        "description": "Транзакции и баланс.",
    },
    {
        "name": "Analytics",
        "description": "Отчетность и графики.",
    },
    {
        "name": "System",
        "description": "Системные ручки.",
    },
]

app = FastAPI(
    title="Center-Invest AI Banking API",
    description="""
### Документация API

Система интеллектуального анализа личных финансов.

    """,
    version="1.0.0",
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    await init_db()
    MLLoader.load()
    async with AsyncSessionLocal() as session:
        await seed_data(session)

app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/health", tags=["System"])
def health():
    return {
        "status": "operational",
        "mode": "demo_frozen_time",
        "system_time": settings.MOCK_NOW
    }