# Multimodal Customer Sentiment Analysis System



An AI-powered customer sentiment analysis system that analyzes customer reviews using text, images, or both.



This project compares three deep learning approaches:

- Text-only sentiment analysis

- Image-only sentiment analysis

- Multimodal sentiment analysis



The system includes a React frontend, FastAPI backend, PostgreSQL database, and an interactive dashboard for analyzing and tracking customer sentiment.



## Features



- Analyze sentiment from customer review text.

- Analyze sentiment from uploaded images.

- Perform multimodal sentiment analysis using text and images.

- Compare three deep learning models.

- Predict five sentiment categories.

- Display prediction confidence and class probabilities.

- Store prediction history in PostgreSQL.

- View dashboard statistics and sentiment visualizations.

- Access prediction endpoints through FastAPI.



## Sentiment Categories



| Class | Sentiment |

|---|---|

| 0 | Very Dissatisfied |

| 1 | Dissatisfied |

| 2 | Neutral |

| 3 | Satisfied |

| 4 | Very Satisfied |



## Technology Stack



| Component | Technologies |

|---|---|

| Programming Language | Python |

| Deep Learning | PyTorch |

| Text Encoder | DistilBERT |

| Image Encoder | ResNet18 |

| NLP | Hugging Face Transformers |

| Backend | FastAPI |

| Database | PostgreSQL |

| ORM | SQLAlchemy |

| Frontend | React, Vite |

| Styling | Tailwind CSS |

| Charts | Recharts |

| API Communication | Axios |

| GPU Acceleration | CUDA |



## Model Architecture



### 1. Text-only Model



- Uses DistilBERT (`distilbert-base-uncased`).

- Extracts contextual features from customer reviews.

- Uses masked mean pooling.

- Produces a 768-dimensional feature representation.

- Uses a classification head to predict five sentiment classes.



### 2. Image-only Model



- Uses ResNet18 as the image encoder.

- Extracts 512-dimensional visual features.

- Uses a classification head to predict five sentiment classes.



### 3. Multimodal Model



The multimodal model combines textual and visual features.



1\. DistilBERT extracts 768-dimensional text features.

2\. ResNet18 extracts 512-dimensional image features.

3\. Both representations are concatenated into a 1,280-dimensional vector.

4\. A fusion classifier processes the combined features.

5\. The model produces five sentiment logits.



**Fusion classifier:**



Linear (1280 â†’ 512) â†’ ReLU â†’ Dropout (0.3) â†’ Linear (512 â†’ 5)


## Dataset

The project uses the **Amazon Reviews 2023 Electronics dataset**.

An expanded dataset was collected and processed to create review-image pairs.

### Dataset Statistics

| Information | Count |
|---|---:|
| Initially collected review-image pairs | 10,000 |
| Valid unique pairs after cleaning | 9,105 |
| Duplicate image contents removed | 78 |
| Missing images removed | 817 |
| Unique products | 8,299 |

### Dataset Splitting

A product-aware splitting strategy was used to reduce product overlap between training, validation, and testing datasets.

| Split | Samples |
|---|---:|
| Training | 7,283 |
| Validation | 911 |
| Testing | 911 |
| **Total** | **9,105** |

The dataset has an imbalanced sentiment distribution, with Very Satisfied reviews forming the largest category.

## Model Evaluation

The following results were obtained on the expanded test dataset.

| Metric | Text-only | Image-only | Multimodal |
|---|---:|---:|---:|
| Accuracy | 74.86% | 56.09% | 73.22% |
| Macro F1 | 51.95% | 22.23% | **53.94%** |
| Weighted F1 | **74.16%** | 53.41% | 74.11% |
| MAE | **0.2931** | 0.9407 | 0.3359 |
| QWK | **0.8575** | 0.1834 | 0.8120 |

### Key Observations

- The text-only model achieved the highest accuracy of 74.86%.
- The multimodal model achieved the highest Macro F1 score of 53.94%.
- The image-only model performed lower across the reported metrics.
- The multimodal model showed a small improvement in Macro F1 over the text-only model.
- Class imbalance remains an important challenge, particularly for minority sentiment categories.

These results suggest that combining text and image features can improve macro-level class performance, although the multimodal model does not outperform the text-only model on every metric.

Detailed results are available in `ml/evaluation/FINAL_MODEL_EVALUATION.md`.

## Training Configuration

The models were trained using PyTorch with GPU acceleration.

| Configuration | Value |
|---|---|
| Text encoder | DistilBERT |
| Image encoder | ResNet18 |
| Maximum text length | 256 tokens |
| Batch size | 4 |
| Mixed precision | FP16 |
| Early stopping | Used |
| Hardware | NVIDIA GPU (RTX 4050) |

The best checkpoints were selected using validation Macro F1.

## Important Note About Model Files

Trained model checkpoints are excluded from GitHub because of their large file sizes.

The application expects the following files inside `ml/saved_models/expanded/`:

- `best_multimodal_model_expanded.pth`
- `best_text_only_model_expanded.pth`
- `best_image_only_model_expanded.pth`

These files must be made available locally before running model inference.


multi-modal_sentiment_analysis_system/
│
├── backend/
│   ├── app/
│   │   ├── database/
│   │   ├── routes/
│   │   ├── schemas/
│   │   ├── services/
│   │   └── utils/
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── services/
│   └── package.json
│
├── ml/
│   ├── evaluation/
│   └── src/
│       └── models/
│
├── docs/
│   ├── screenshots/
│   ├── API_DOCUMENTATION.md
│   ├── DATASET.md
│   ├── LIMITATIONS.md
│   ├── MODEL_DOCUMENTATION.md
│   ├── PROJECT_OVERVIEW.md
│   └── SYSTEM_ARCHITECTURE.md
│
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt

## Installation and Setup

### Prerequisites

- Python 3.11
- Node.js and npm
- PostgreSQL
- Git
- NVIDIA GPU with compatible CUDA support (recommended)

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd multi-modal_sentiment_analysis_system
```

### 2. Create a Python Virtual Environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Python Dependencies

```powershell
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root.

```env
DATABASE_URL=postgresql://USERNAME:PASSWORD@localhost:5432/multimodal_sentiment_db
```

Replace the placeholders with your local PostgreSQL credentials.

**Never upload your actual `.env` file to GitHub.**

### 5. Create the PostgreSQL Database

Run the following SQL command in PostgreSQL:

```sql
CREATE DATABASE multimodal_sentiment_db;
```

Initialize the database tables using the project's database initialization module.

### 6. Add Model Checkpoints

Place the three expanded-dataset checkpoints inside:

```text
ml/saved_models/expanded/
```

Required files:

```text
best_multimodal_model_expanded.pth
best_text_only_model_expanded.pth
best_image_only_model_expanded.pth
```

The model loader expects these exact filenames.

### 7. Start the Backend

From the project root:

```powershell
uvicorn app.main:app --reload --app-dir backend
```

Backend:

`http://localhost:8000`

FastAPI Swagger documentation:

`http://localhost:8000/docs`

### 8. Start the Frontend

Open another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Frontend:

`http://localhost:5173`

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/predictions/predict` | Generate sentiment prediction |
| GET | `/api/v1/predictions/history` | Retrieve prediction history |
| GET | `/api/v1/statistics/dashboard` | Retrieve dashboard statistics |
| GET | `/` | Backend status |
| GET | `/health` | Health check |

For detailed request and response formats, see [API Documentation](docs/API_DOCUMENTATION.md).

## Application Screenshots

Screenshots will be added to `docs/screenshots/`.

Planned screenshots:

1. Dashboard overview
2. Text-only sentiment analysis
3. Image-only sentiment analysis
4. Multimodal sentiment analysis
5. Prediction history
6. FastAPI Swagger documentation

## Limitations

- The dataset contains an imbalanced distribution of sentiment classes.
- Image-only prediction currently performs worse than text-based prediction.
- Prediction confidence does not guarantee correctness.
- The system is limited to the five sentiment categories used during training.
- Model checkpoints are not included in the GitHub repository.

## Future Improvements

- Improve minority-class sentiment recognition.
- Explore alternative multimodal fusion techniques.
- Improve the contribution of visual features.
- Evaluate additional pretrained architectures.
- Expand the dataset with more balanced sentiment categories.
- Improve model interpretability.

## Detailed Documentation

| Document | Description |
|---|---|
| [Project Overview](docs/PROJECT_OVERVIEW.md) | Objectives, scope, and features |
| [System Architecture](docs/SYSTEM_ARCHITECTURE.md) | System components and data flow |
| [Model Documentation](docs/MODEL_DOCUMENTATION.md) | Deep learning architectures |
| [Dataset Documentation](docs/DATASET.md) | Dataset collection and preprocessing |
| [API Documentation](docs/API_DOCUMENTATION.md) | Backend API details |
| [Limitations](docs/LIMITATIONS.md) | Current limitations and improvements |
| [Model Evaluation](ml/evaluation/FINAL_MODEL_EVALUATION.md) | Detailed evaluation results |

## Author

Developed as an academic project exploring deep learning, natural language processing, computer vision, and multimodal sentiment analysis.

---

*This project is intended for educational and research purposes.*
