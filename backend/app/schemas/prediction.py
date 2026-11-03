from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    sentiment: str
    predicted_class: int = Field(ge=0, le=4)
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]
    model_used: str
    image_used: bool