\# System Architecture



\## 1. Overview



The Multimodal Customer Sentiment Analysis System follows a modular architecture consisting of three main layers:



1\. Machine Learning Layer

2\. Backend Layer

3\. Frontend Layer



PostgreSQL is used to store prediction records and support dashboard statistics.



\## 2. Architecture Components



\### 2.1 Machine Learning Layer



The machine learning layer contains three independently trained models.



\*\*Text-only model\*\*

\- Uses DistilBERT to extract contextual features from review text.

\- Uses a classification head to predict one of five sentiment classes.



\*\*Image-only model\*\*

\- Uses ResNet18 to extract visual features from product images.

\- Uses a classification head to predict sentiment.



\*\*Multimodal model\*\*

\- Extracts text features using DistilBERT.

\- Extracts image features using ResNet18.

\- Combines the 768-dimensional text representation and 512-dimensional image representation.

\- Passes the concatenated 1280-dimensional representation through a fusion classifier.



\### 2.2 Backend Layer



The backend is developed using FastAPI.



Its main responsibilities include:



\- Receiving prediction requests from the frontend.

\- Validating text and image inputs.

\- Loading the trained model checkpoints.

\- Selecting the appropriate model based on the available input.

\- Generating sentiment predictions and class probabilities.

\- Saving prediction records in PostgreSQL.

\- Providing prediction history and dashboard statistics through API endpoints.



\### 2.3 Database Layer



PostgreSQL stores prediction records.



The prediction records include information such as:



\- Review title

\- Review text

\- Image path

\- Predicted sentiment

\- Predicted class

\- Confidence score

\- Class probabilities

\- Model used

\- Whether an image was used

\- Prediction timestamp



\### 2.4 Frontend Layer



The frontend is developed using React and Vite.



It provides the following interfaces:



\- \*\*Dashboard:\*\* Displays sentiment statistics, charts, and recent predictions.

\- \*\*Analyze:\*\* Allows users to submit review text and images for sentiment analysis.

\- \*\*History:\*\* Displays previously stored predictions.



The frontend communicates with the backend through HTTP API requests.



\## 3. Prediction Workflow



1\. The user opens the web application.

2\. The user enters review information and optionally uploads an image.

3\. The frontend sends the input to the FastAPI backend.

4\. The backend validates the submitted information.

5\. The prediction service selects the appropriate model:

&#x20;  - Text-only model when only text is provided.

&#x20;  - Image-only model when only an image is provided.

&#x20;  - Multimodal model when both text and image are provided.

6\. The selected model generates a sentiment prediction.

7\. The backend prepares the predicted class, confidence, and class probabilities.

8\. The prediction record is stored in PostgreSQL.

9\. The backend returns the prediction result to the frontend.

10\. The frontend displays the result to the user.



\## 4. High-Level Data Flow



```text

&#x20;            USER

&#x20;              |

&#x20;              v

&#x20;       REACT FRONTEND

&#x20;              |

&#x20;              v

&#x20;        FASTAPI BACKEND

&#x20;              |

&#x20;              v

&#x20;      INPUT VALIDATION

&#x20;              |

&#x20;              v

&#x20;      MODEL SELECTION

&#x20;              |

&#x20;      +-------+-------+

&#x20;      |       |       |

&#x20;      v       v       v

&#x20;     TEXT   IMAGE   MULTIMODAL

&#x20;     MODEL  MODEL     MODEL

&#x20;      |       |       |

&#x20;      +-------+-------+

&#x20;              |

&#x20;              v

&#x20;      SENTIMENT RESULT

&#x20;              |

&#x20;         +----+----+

&#x20;         |         |

&#x20;         v         v

&#x20;     POSTGRESQL  FRONTEND

&#x20;     DATABASE    RESPONSE

```



\## 5. API Communication



The frontend communicates with the backend using REST API endpoints.



| Method | Endpoint | Purpose |

|---|---|---|

| POST | `/api/v1/predictions/predict` | Generate a sentiment prediction |

| GET | `/api/v1/predictions/history` | Retrieve prediction history |

| GET | `/api/v1/statistics/dashboard` | Retrieve dashboard statistics |



\## 6. Technology Summary



| Component | Technology |

|---|---|

| Frontend | React, Vite |

| Backend | FastAPI |

| Database | PostgreSQL |

| ORM | SQLAlchemy |

| Text Encoder | DistilBERT |

| Image Encoder | ResNet18 |

| Deep Learning Framework | PyTorch |

| API Testing | Postman / Swagger UI |



\## 7. Architecture Summary



The architecture separates model inference, API handling, database operations, and user interface responsibilities.



This modular design makes the system easier to maintain, test, and extend.