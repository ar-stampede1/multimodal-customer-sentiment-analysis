\# API Documentation



\## 1. Introduction



The backend of the Multimodal Customer Sentiment Analysis System is developed using FastAPI.



It provides REST API endpoints for sentiment prediction, prediction history, and dashboard statistics.



Base URL:



```text

http://localhost:8000

```



API version prefix:



```text

/api/v1

```



Interactive API documentation is available through Swagger UI:



```text

http://localhost:8000/docs

```



\## 2. API Endpoints



| Method | Endpoint | Description |

|---|---|---|

| POST | `/api/v1/predictions/predict` | Generate a sentiment prediction |

| GET | `/api/v1/predictions/history` | Retrieve prediction history |

| GET | `/api/v1/statistics/dashboard` | Retrieve dashboard statistics |

| GET | `/` | Backend status |

| GET | `/health` | Health check |



\## 3. Sentiment Prediction



\### Endpoint



```http

POST /api/v1/predictions/predict

```



\### Description



Accepts customer review information and generates a sentiment prediction using the appropriate trained model.



\### Request Format



The endpoint accepts `multipart/form-data`.



| Parameter | Type | Required | Description |

|---|---|---|---|

| review\_title | String | No | Customer review title |

| review\_text | String | Conditional | Customer review text |

| image | File | Conditional | Uploaded product image |



At least one valid input modality must be provided.



Supported image formats include JPEG, PNG, and WebP. The maximum image size is 5 MB.



\### Model Selection



| Available Input | Model Used |

|---|---|

| Text only | Text-only model |

| Image only | Image-only model |

| Text and image | Multimodal model |



\### Prediction Response



The response includes prediction information such as:



\- Predicted sentiment

\- Predicted class

\- Confidence score

\- Class probabilities

\- Model used



The prediction is also stored in the PostgreSQL database.



\## 4. Prediction History



\### Endpoint



```http

GET /api/v1/predictions/history

```



\### Description



Retrieves previously stored sentiment predictions.



\### Query Parameters



| Parameter | Description |

|---|---|

| limit | Maximum number of records to retrieve |

| offset | Number of records to skip |



Pagination allows the frontend to retrieve prediction records in smaller groups.



\## 5. Dashboard Statistics



\### Endpoint



```http

GET /api/v1/statistics/dashboard

```



\### Description



Returns aggregated prediction statistics used by the dashboard.



The frontend uses this endpoint to display sentiment statistics and visualizations.



\## 6. Health Check



\### Endpoint



```http

GET /health

```



\### Description



Provides a health-check route to verify that the backend service is responding.



\## 7. Root Endpoint



\### Endpoint



```http

GET /

```



\### Description



Provides the backend's root status response.



\## 8. Database Integration



The backend uses SQLAlchemy to communicate with PostgreSQL.



Prediction records contain fields including:



\- Review title and review text

\- Image path

\- Sentiment and predicted class

\- Confidence and class probabilities

\- Model used

\- Image usage indicator

\- Creation timestamp



Database connection settings are loaded from the project's `.env` file.



\## 9. Testing



The API can be tested using:



\- FastAPI Swagger UI

\- Postman

\- Python test scripts



Swagger UI:



`http://localhost:8000/docs`



\## 10. Error Handling



The backend validates incoming prediction inputs and image files before model inference.



Invalid inputs, unsupported image formats, oversized images, and server-side failures may result in HTTP error responses.



Refer to the running Swagger documentation for the exact response schemas and status codes.