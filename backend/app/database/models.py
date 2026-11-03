
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    JSON,
)

from app.database.connection import Base


class Prediction(Base):

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)

    review_title = Column(String(500), nullable=True)

    review_text = Column(String, nullable=True)

    image_path = Column(String(1000), nullable=True)

    sentiment = Column(String(50), nullable=False)

    predicted_class = Column(Integer, nullable=False)

    confidence = Column(Float, nullable=False)

    probabilities = Column(JSON, nullable=False)

    model_used = Column(String(30), nullable=False)

    image_used = Column(Boolean, default=False, nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
