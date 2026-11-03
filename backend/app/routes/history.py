
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.database.dependencies import get_db
from app.database.models import Prediction


router = APIRouter(
    prefix="/api/v1/predictions",
    tags=["Prediction History"],
)


@router.get("/history")
def get_prediction_history(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):

    try:
        predictions = (
            db.query(Prediction)
            .order_by(desc(Prediction.created_at))
            .offset(offset)
            .limit(limit)
            .all()
        )

        total = db.query(Prediction).count()

        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "predictions": [
                {
                    "id": prediction.id,
                    "review_title": prediction.review_title,
                    "review_text": prediction.review_text,
                    "sentiment": prediction.sentiment,
                    "predicted_class": prediction.predicted_class,
                    "confidence": prediction.confidence,
                    "probabilities": prediction.probabilities,
                    "model_used": prediction.model_used,
                    "image_used": prediction.image_used,
                    "created_at": prediction.created_at,
                }
                for prediction in predictions
            ],
        }

    except Exception as error:
        print("History retrieval error:", repr(error))

        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve prediction history.",
        )
