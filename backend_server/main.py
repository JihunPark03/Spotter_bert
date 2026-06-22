# ──────────────────────────────────────────────────────────────
# main.py   (Spotter API v0.3.0)
# ──────────────────────────────────────────────────────────────
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging, os
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session

from services.detect_service import detect_ad as detect_service
from services.feedback_service import FeedbackService
from repositories.feedback_repository import FeedbackRepository
from db_init import get_db, create_tables



# ──────────────── Basic Settings ────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Create tables on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Creating database tables...")
    create_tables()
    yield  # Allow FastAPI to start
    print("FastAPI is shutting down...")  # Shutdown logic (optional)
    

app = FastAPI(
    title="Spotter API",
    description="Backend API for the Spotter Chrome Extension",
    version="0.3.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


# ──────────────── Pydantic Models ────────────────
class AdDetectRequest(BaseModel):
    text: str

class AdDetectResponse(BaseModel):
    prob_ad: float
    is_ad: bool
    cached: bool

class FeedbackRequest(BaseModel):
    text: str
    is_ad: bool

# ──────────────── Endpoints ────────────────
@app.get("/")
async def root():
    return {"status": "online", "message": "Spotter API is running"}

@app.post("/detect-ad", response_model=AdDetectResponse)
async def detect_ad(req: AdDetectRequest):
    text = req.text.strip()
    if not text:
        return JSONResponse(status_code=400, content={"detail": "Empty text"})

    return detect_service(text)

@app.post("/feedback")
async def save_feedback(req: FeedbackRequest, db: Session = Depends(get_db)):
    service = FeedbackService(FeedbackRepository(db))
    try:
        fb = service.save_feedback(req.text, req.is_ad)
        return {"status": "ok", "id": fb.id}
    except ValueError as ve:
        return JSONResponse(status_code=400, content={"detail": str(ve)})
    except Exception:
        logger.exception("Failed to save feedback")
        db.rollback()
        return JSONResponse(
            status_code=500,
            content={"detail": "Failed to save feedback"},
        )

@app.post("/recommendations")
async def get_recommendations():
    return JSONResponse(
        status_code=501,
        content={"detail": "Recommendation system disabled in this deployment."}
    )

# ──────────────── Local Run ────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )
