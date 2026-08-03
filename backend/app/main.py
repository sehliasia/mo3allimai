from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import auth, health
from app.core.config import get_settings

settings = get_settings()
app = FastAPI(title="Mo3allimAI API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=[settings.frontend_url], allow_credentials=True, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["Authorization", "Content-Type"])
app.include_router(health.router, prefix="/api")
app.include_router(auth.router, prefix="/api")
