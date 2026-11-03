
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database.dependencies import get_db
from app.database.models import Prediction


router = APIRouter(
    prefix="/api/v1/statistics",
    tags=["Dashboard Statistics"],
)


@router.get("/dashboard")
def get_dashboard_statistics(
    db: Session = Depends(get_db),
):

    try:
        # Total number of predictions
        total_predictions = db.query(Prediction).count()

        # Count predictions by sentiment
        sentiment_rows = (
            db.query(
                Prediction.sentiment,
                func.count(Prediction.id),
            )
            .group_by(Prediction.sentiment)
            .all()
        )

        sentiment_counts = {
            "Very Dissatisfied": 0,
            "Dissatisfied": 0,
            "Neutral": 0,
            "Satisfied": 0,
            "Very Satisfied": 0,
        }

        for sentiment, count in sentiment_rows:
            sentiment_counts[sentiment] = count

        # Sentiment percentages
        sentiment_percentages = {
            sentiment: round(
                (count / total_predictions) * 100, 2
            ) if total_predictions else 0
            for sentiment, count in sentiment_counts.items()
        }

        # Count predictions by model
        model_rows = (
            db.query(
                Prediction.model_used,
                func.count(Prediction.id),
            )
            .group_by(Prediction.model_used)
            .all()
        )

        model_usage = {
            "text_only": 0,
            "image_only": 0,
            "multimodal": 0,
        }

        for model, count in model_rows:
            model_usage[model] = count

        # Average confidence
        average_confidence = (
            db.query(func.avg(Prediction.confidence)).scalar() or 0
        )

        return {
            "total_predictions": total_predictions,
            "sentiment_distribution": {
                "counts": sentiment_counts,
                "percentages": sentiment_percentages,
            },
            "model_usage": model_usage,
            "average_confidence": round(
                float(average_confidence), 4
            ),
        }

    except Exception as error:
        print("Dashboard statistics error:", repr(error))

        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve dashboard statistics.",
        )
