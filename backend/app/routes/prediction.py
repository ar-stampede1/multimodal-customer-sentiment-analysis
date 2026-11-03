
from io import BytesIO

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Depends
from PIL import Image, UnidentifiedImageError
from sqlalchemy.orm import Session

from app.schemas.prediction import PredictionResponse
from app.services.model_loader import load_models
from app.services.prediction_service import PredictionService
from app.database.connection import get_db
from app.database.models import Prediction


router = APIRouter(
    prefix="/api/v1/predictions",
    tags=["Sentiment Prediction"],
)


# LOAD MODELS ONCE
print("Initializing prediction service...")

models = load_models()
prediction_service = PredictionService(models)

print("Prediction service ready!")


# PREDICTION ENDPOINT
@router.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(
    review_title: str = Form(""),
    review_text: str = Form(""),
    image: UploadFile | None = File(None),
    db: Session = Depends(get_db),
):

    # Validate input
    if not review_title.strip() and not review_text.strip() and image is None:
        raise HTTPException(
            status_code=400,
            detail="Provide review text, an image, or both.",
        )

    pil_image = None

    # Process uploaded image only if provided
    if image is not None:

        allowed_types = {
            "image/jpeg",
            "image/png",
            "image/webp",
        }

        if image.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail="Only JPEG, PNG and WebP images are supported.",
            )

        try:
            image_bytes = await image.read()

            if not image_bytes:
                raise HTTPException(
                    status_code=400,
                    detail="Uploaded image is empty.",
                )

            if len(image_bytes) > 5 * 1024 * 1024:
                raise HTTPException(
                    status_code=413,
                    detail="Image size must not exceed 5 MB.",
                )

            pil_image = Image.open(BytesIO(image_bytes))
            pil_image.load()
            pil_image = pil_image.convert("RGB")

        except HTTPException:
            raise

        except (UnidentifiedImageError, OSError, ValueError):
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is not a valid image.",
            )

        finally:
            await image.close()

    # Run model inference (works for text, image, or both)
    try:

        result = prediction_service.predict(
            review_title=review_title,
            review_text=review_text,
            image=pil_image,
        )

        # Save prediction in PostgreSQL
        prediction_record = Prediction(
            review_title=review_title or None,
            review_text=review_text or None,
            sentiment=result["sentiment"],
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            model_used=result["model_used"],
            image_used=result["image_used"],
        )

        db.add(prediction_record)
        db.commit()
        db.refresh(prediction_record)

        return PredictionResponse(**result)

    except ValueError as error:
        db.rollback()

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:
        db.rollback()

        print("Prediction error:", repr(error))

        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during prediction.",
        )
