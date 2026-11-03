\# Project Overview



\## 1. Introduction



The Multimodal Customer Sentiment Analysis System is an AI-powered application designed to analyze customer sentiment from product reviews using textual and visual information.



The system uses three deep learning approaches: text-only analysis, image-only analysis, and multimodal analysis. It allows users to analyze reviews, view prediction confidence, and track previous predictions through an interactive dashboard.



\## 2. Problem Statement



Customer reviews contain useful information about product satisfaction. However, analyzing large amounts of review data manually can be time-consuming.



Text alone may not always capture all the information available in a review. Product images can provide additional visual context.



This project explores the use of natural language processing and computer vision to classify customer sentiment using text, images, or both.



\## 3. Project Objectives



\- Develop a text-based sentiment classification model.

\- Develop an image-based sentiment classification model.

\- Build a multimodal model that combines textual and visual features.

\- Compare the performance of all three models.

\- Classify customer reviews into five sentiment categories.

\- Develop a web application for submitting reviews and viewing predictions.

\- Store prediction records in a PostgreSQL database.

\- Display sentiment statistics and prediction history through a dashboard.



\## 4. Scope



The system supports three prediction modes:



1\. \*\*Text-only:\*\* Uses the customer review text.

2\. \*\*Image-only:\*\* Uses an uploaded product image.

3\. \*\*Multimodal:\*\* Uses both review text and an uploaded image.



The system predicts one of five sentiment categories, ranging from Very Dissatisfied to Very Satisfied.



\## 5. Sentiment Categories



| Class | Sentiment |

|---|---|

| 0 | Very Dissatisfied |

| 1 | Dissatisfied |

| 2 | Neutral |

| 3 | Satisfied |

| 4 | Very Satisfied |



\## 6. Main Technologies



\- Python and PyTorch

\- DistilBERT for text feature extraction

\- ResNet18 for image feature extraction

\- FastAPI for backend development

\- React and Vite for frontend development

\- PostgreSQL for prediction storage



\## 7. Expected Outcome



The project provides a web-based interface where users can submit customer reviews and receive sentiment predictions.



It also provides a comparison of text-only, image-only, and multimodal approaches, along with prediction history and dashboard statistics.



The evaluation results are documented separately in the model evaluation report.