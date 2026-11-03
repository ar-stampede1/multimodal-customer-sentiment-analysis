
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.prediction import router as prediction_router
from app.routes.history import router as history_router
from app.routes.statistics import router as statistics_router

app = FastAPI(
    title="Multimodal Customer Sentiment Analysis API",
    description=(
        "An API for analyzing customer sentiment using "
        "review text and customer-uploaded images."
    ),
    version="1.0.0",
)

# Allow the React frontend to communicate with the backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction_router)
app.include_router(history_router)
app.include_router(statistics_router)

@app.get("/")
def root():
    return {
        "message": "Multimodal Sentiment Analysis API is running!",
        "status": "healthy",
        "version": "1.0.0",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "sentiment-analysis-backend",
    }
