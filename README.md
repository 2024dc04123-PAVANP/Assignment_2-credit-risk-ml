# Credit Card Default Prediction using Machine Learning



---

## a. Problem Statement

Credit card default prediction is a critical problem in the financial domain, as incorrect assessment of customer credit risk can result in significant financial losses. The objective of this project is to build and evaluate multiple machine learning classification models that can predict whether a customer will default on their next credit card payment based on demographic information, repayment history, bill amounts, and previous payment behavior.

The problem is formulated as a binary classification task, where:
- 0 → No Default
- 1 → Default

---

## b. Dataset Description  [1 Mark]

- Dataset Name: Credit Card Default Dataset  
- Source: Kaggle / UCI Machine Learning Repository  
- Total Records: 30,000  
- Number of Input Features: 23  
- Target Variable: `default.payment.next.month`  

### Feature Categories
- Demographic Features: Gender, Education, Marital Status, Age  
- Financial Features: Credit limit and bill amounts  
- Behavioral Features: Past repayment status and payment history  

The dataset is imbalanced, with a larger number of non-defaulters compared to defaulters. Therefore, relying solely on accuracy is insufficient, and multiple evaluation metrics are required.

---

## c. Models Used and Evaluation Metrics  [6 Marks]

The following machine learning models were implemented and evaluated:

- Logistic Regression  
- Decision Tree  
- K-Nearest Neighbors (KNN)  
- Naive Bayes  
- Random Forest (Ensemble)  
- XGBoost (Ensemble)  

### Evaluation Metrics Used
- Accuracy  
- AUC (Area Under ROC Curve)  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

---

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.808 | 0.708 | 0.687 | 0.240 | 0.355 | 0.324 |
| Decision Tree | 0.811 | 0.722 | 0.626 | 0.356 | 0.454 | 0.371 |
| KNN | 0.793 | 0.701 | 0.549 | 0.356 | 0.432 | 0.323 |
| Naive Bayes | 0.416 | 0.652 | 0.250 | 0.818 | 0.382 | 0.111 |
| Random Forest (Ensemble) | 0.817 | 0.769 | 0.663 | 0.354 | 0.462 | 0.390 |
| XGBoost (Ensemble) | 0.819 | 0.778 | 0.666 | 0.363 | 0.470 | 0.397 |

---

## d. Observations on Model Performance  [3 Marks]

| ML Model Name | Observation about Model Performance |
|--------------|-------------------------------------|
| Logistic Regression | Logistic Regression achieved good accuracy and precision but very low recall, indicating poor identification of actual defaulters. It serves as a strong baseline model but struggles with class imbalance. |
| Decision Tree | Decision Tree improved recall and F1 score by capturing non-linear patterns in the data, but it may suffer from overfitting. |
| KNN | KNN showed balanced recall and F1 score but slightly lower accuracy. Its performance is sensitive to feature scaling and the choice of k value. |
| Naive Bayes | Naive Bayes achieved very high recall but extremely low precision and accuracy, resulting in many false positives due to strong independence assumptions. |
| Random Forest (Ensemble) | Random Forest demonstrated strong overall performance with improved AUC, F1 score, and MCC, benefiting from ensemble learning and reduced overfitting. |
| XGBoost (Ensemble) | XGBoost delivered the best overall performance with the highest AUC, F1 score, and MCC, making it the most effective model for this dataset. |

---

## Student Details

- Name: PAVAN P  
- BITS ID: 2024DC04123  

---

## Conclusion

This project demonstrates the effectiveness of applying machine learning techniques to credit risk prediction. While simpler models such as Logistic Regression provide interpretability, ensemble models like Random Forest and XGBoost outperform other models across most evaluation metrics. These results highlight the importance of using robust evaluation measures and ensemble learning methods when dealing with imbalanced financial datasets.
