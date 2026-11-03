\# Model Documentation



\## 1. Introduction



The system implements three deep learning models for customer sentiment classification:



1\. Text-only model

2\. Image-only model

3\. Multimodal fusion model



All three models classify customer sentiment into five categories.



\## 2. Text-only Model



\### Architecture: DistilBERT



The text-only model uses DistilBERT, a pretrained transformer-based language model.



\*\*Working:\*\*



1\. The review text is tokenized using the DistilBERT tokenizer.

2\. Token IDs and attention masks are passed to DistilBERT.

3\. The model generates contextual token representations.

4\. Masked mean pooling combines token representations into a 768-dimensional feature vector.

5\. The feature vector passes through a classification head.

6\. The model predicts one of five sentiment categories.



\*\*Classification head:\*\*



```text

Input: 768 features

&#x20;      |

Linear (768 → 512)

&#x20;      |

ReLU

&#x20;      |

Dropout (0.3)

&#x20;      |

Linear (512 → 5)

&#x20;      |

Sentiment Prediction

```



\## 3. Image-only Model



\### Architecture: ResNet18



The image-only model uses ResNet18, a convolutional neural network for image feature extraction.



\*\*Working:\*\*



1\. The uploaded product image is preprocessed.

2\. The image is passed through the ResNet18-based image encoder.

3\. The encoder extracts a 512-dimensional visual feature representation.

4\. The feature vector passes through a classification head.

5\. The model predicts one of five sentiment categories.



\*\*Classification head:\*\*



```text

Input: 512 features

&#x20;      |

Linear (512 → 256)

&#x20;      |

ReLU

&#x20;      |

Dropout (0.3)

&#x20;      |

Linear (256 → 5)

&#x20;      |

Sentiment Prediction

```



\## 4. Multimodal Fusion Model



The multimodal model combines information from both text and images.



\*\*Text branch:\*\*

\- DistilBERT extracts contextual information from the customer review.

\- Masked mean pooling produces a 768-dimensional feature vector.



\*\*Image branch:\*\*

\- ResNet18 extracts visual information from the uploaded image.

\- The image encoder produces a 512-dimensional feature vector.



\*\*Feature fusion:\*\*



The two feature vectors are concatenated to create a 1280-dimensional representation.



```text

Text Features: 768 dimensions

Image Features: 512 dimensions

&#x20;                |

&#x20;                v

&#x20;      Concatenation

&#x20;                |

&#x20;                v

&#x20;     1280-dimensional vector

&#x20;                |

&#x20;                v

&#x20;      Linear (1280 → 512)

&#x20;                |

&#x20;                v

&#x20;              ReLU

&#x20;                |

&#x20;                v

&#x20;          Dropout (0.3)

&#x20;                |

&#x20;                v

&#x20;        Linear (512 → 5)

&#x20;                |

&#x20;                v

&#x20;      Sentiment Prediction

```



\## 5. Sentiment Classification



| Class Index | Sentiment |

|---|---|

| 0 | Very Dissatisfied |

| 1 | Dissatisfied |

| 2 | Neutral |

| 3 | Satisfied |

| 4 | Very Satisfied |



The model produces class scores that are converted into probabilities for the prediction response.



\## 6. Model Training Configuration



| Parameter | Configuration |

|---|---|

| Text encoder | DistilBERT |

| Image encoder | ResNet18 |

| Maximum text length | 256 tokens |

| Batch size | 4 |

| Mixed precision | Automatic Mixed Precision (FP16) |

| Early stopping | Enabled |

| Training GPU | NVIDIA RTX 4050 Laptop GPU |



\## 7. Model Checkpoints



The backend loads trained model checkpoints from:



`ml/saved\_models/expanded/`



Expected checkpoint files:



\- `best\_multimodal\_model\_expanded.pth`

\- `best\_text\_only\_model\_expanded.pth`

\- `best\_image\_only\_model\_expanded.pth`



Each checkpoint contains the trained model state used for inference.



\## 8. Model Evaluation Summary



The following results were obtained on the expanded test dataset.



| Metric | Text-only | Image-only | Multimodal |

|---|---:|---:|---:|

| Accuracy | 74.86% | 56.09% | 73.22% |

| Macro F1 | 51.95% | 22.23% | 53.94% |

| Weighted F1 | 74.16% | 53.41% | 74.11% |

| MAE | 0.2931 | 0.9407 | 0.3359 |

| QWK | 0.8575 | 0.1834 | 0.8120 |



\### Observations



\- The text-only model achieved the highest accuracy.

\- The multimodal model achieved the highest macro F1 score.

\- The image-only model performed lower than the other two models.

\- The multimodal model showed a small improvement in macro F1 compared with the text-only model.

\- No single model performed best across every evaluation metric.



\## 9. Conclusion



The project compares three approaches to sentiment classification using textual, visual, and combined information.



The evaluation demonstrates that combining text and image features can improve macro F1 performance, although the text-only model achieved higher accuracy on the expanded test dataset.