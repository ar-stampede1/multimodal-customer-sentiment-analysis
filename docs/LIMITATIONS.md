\# Limitations and Future Improvements



\## 1. Introduction



The Multimodal Customer Sentiment Analysis System demonstrates sentiment classification using textual and visual information.



Although the system provides three prediction approaches, there are some limitations related to dataset quality, model performance, and multimodal learning.



\## 2. Current Limitations



\### 2.1 Dataset Imbalance



The dataset contains a significantly larger number of five-star reviews compared with other rating categories.



This imbalance can affect the model's ability to recognize minority sentiment classes.



\### 2.2 Image-only Model Performance



The image-only model performs worse than the text-only and multimodal models on the expanded test dataset.



Product images do not always contain enough information to determine customer satisfaction.



\### 2.3 Multimodal Performance



The multimodal model does not outperform the text-only model on every evaluation metric.



Although it achieves a higher macro F1 score, its accuracy and quadratic weighted kappa are lower than those of the text-only model.



\### 2.4 Limited Dataset Category



The dataset focuses on Amazon Electronics products.



The model's performance on other product categories has not been established.



\### 2.5 Prediction Reliability



The confidence score represents the model's confidence in an individual prediction. It does not guarantee that the prediction is correct.



\### 2.6 Hardware Requirements



Training transformer-based and computer vision models requires considerable computational resources.



GPU acceleration is recommended for efficient model training.



\## 3. Future Improvements



The following improvements could be explored in future versions:



\- Collect more balanced customer review data.

\- Improve minority-class sentiment classification.

\- Experiment with alternative image encoders.

\- Explore attention-based multimodal fusion techniques.

\- Improve the contribution of visual features during fusion.

\- Evaluate additional pretrained transformer architectures.

\- Introduce model explainability techniques.

\- Test the system on additional product categories.

\- Improve prediction calibration and reliability.



\## 4. Conclusion



The project provides a foundation for multimodal customer sentiment analysis.



The evaluation results show that combining textual and visual information can improve macro F1 performance, but multimodal learning does not guarantee better performance across all metrics.



Further experimentation with dataset balancing, feature extraction, and fusion techniques could improve the system's overall performance.