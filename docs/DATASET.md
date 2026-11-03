\# Dataset Documentation



\## 1. Dataset Overview



This project uses the \*\*Amazon Reviews 2023 Electronics dataset\*\* for multimodal customer sentiment analysis.



The dataset contains customer review information and associated product images. It is used to train and evaluate three models: text-only, image-only, and multimodal.



The dataset was accessed using the Hugging Face Datasets platform.



\## 2. Dataset Collection



The expanded dataset collection initially contained 10,000 review-image pairs.



The collection process gathered customer review information and corresponding product images for model training.



\## 3. Data Cleaning and Preprocessing



The collected data was cleaned to improve dataset quality.



The following cleaning operations were performed:



\- Removed duplicate image contents.

\- Removed records with missing images.

\- Retained valid and unique review-image pairs.

\- Prepared the cleaned data for model training and evaluation.



\### Dataset Cleaning Summary



| Description | Count |

|---|---:|

| Initially collected pairs | 10,000 |

| Duplicate image contents removed | 78 |

| Records with missing images removed | 817 |

| Final valid review-image pairs | 9,105 |

| Unique products | 8,299 |



\## 4. Rating Distribution



The cleaned dataset contains the following rating distribution:



| Rating | Number of Reviews |

|---|---:|

| 1 star | 669 |

| 2 stars | 345 |

| 3 stars | 529 |

| 4 stars | 1,398 |

| 5 stars | 6,164 |

| \*\*Total\*\* | \*\*9,105\*\* |



The dataset is imbalanced, with five-star reviews forming the largest category.



\## 5. Sentiment Mapping



The original star ratings are mapped to five sentiment classes.



| Rating | Sentiment Class |

|---|---|

| 1 star | Very Dissatisfied |

| 2 stars | Dissatisfied |

| 3 stars | Neutral |

| 4 stars | Satisfied |

| 5 stars | Very Satisfied |



The resulting class indices are:



| Class Index | Sentiment |

|---|---|

| 0 | Very Dissatisfied |

| 1 | Dissatisfied |

| 2 | Neutral |

| 3 | Satisfied |

| 4 | Very Satisfied |



\## 6. Dataset Splitting



The cleaned dataset was divided into training, validation, and testing sets.



A product-aware splitting strategy was used to reduce data leakage between the different sets.



| Dataset Split | Number of Samples |

|---|---:|

| Training | 7,283 |

| Validation | 911 |

| Testing | 911 |

| \*\*Total\*\* | \*\*9,105\*\* |



The splits were checked for product and image-hash overlap.



No product or image-hash overlap was found between the training, validation, and testing sets.



\## 7. Data Preprocessing



\### Text Preprocessing



\- Review text is tokenized using the DistilBERT tokenizer.

\- The maximum sequence length is 256 tokens.

\- Attention masks are used to identify valid tokens.

\- Masked mean pooling is used to obtain text representations.



\### Image Preprocessing



\- Product images are loaded and prepared for the ResNet18 image encoder.

\- The image encoder extracts a 512-dimensional visual feature representation.



\## 8. Dataset Usage



The dataset supports three model configurations:



1\. \*\*Text-only model:\*\* Uses review text.

2\. \*\*Image-only model:\*\* Uses product images.

3\. \*\*Multimodal model:\*\* Uses both review text and product images.



The same five sentiment categories are used across all three configurations.



\## 9. Dataset Limitations



\- The dataset has a significant imbalance toward five-star reviews.

\- The number of samples in minority sentiment categories is comparatively small.

\- Product images may not always provide direct evidence of customer satisfaction.

\- The dataset is limited to the Amazon Electronics product category.



These limitations may influence model performance, particularly for minority sentiment classes.



\## 10. Dataset Source



Dataset: Amazon Reviews 2023



Category: Electronics



Platform: Hugging Face Datasets



The dataset was accessed using a pinned repository revision to maintain reproducibility.