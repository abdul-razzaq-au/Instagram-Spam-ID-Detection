# Instagram-Spam-ID-Detection
Machine learning–based detection of fake Instagram accounts using user profile and activity metrics. Random Forest was selected as the best-performing model after comparative evaluation.

# 📸 Instagram Fake / Scam Account Detection using Machine Learning

## 📌 Project Overview

The rapid growth of social media platforms has led to a significant rise in fake and scam accounts, posing risks such as misinformation, fraud, and privacy violations. This project focuses on detecting fake Instagram accounts using machine learning classification techniques based on user profile characteristics and activity metrics.

The project follows a complete end-to-end data science pipeline including exploratory data analysis (EDA), feature engineering, feature selection, model building, evaluation, and comparative analysis to identify the most effective model for fake account detection.

---

## 🎯 Objectives

- To analyze Instagram account attributes that differentiate fake and genuine users  
- To engineer meaningful features that improve classification performance  
- To build and compare multiple machine learning models  
- To select the best-performing model using robust evaluation metrics  
- To identify the most important features contributing to fake account detection  

---

## 📂 Dataset Description

The dataset consists of Instagram user profile information divided into separate training and testing datasets. Each observation represents an Instagram account labeled as either **Fake** or **Real**.

### Features include:
- Profile picture availability  
- Username and fullname characteristics  
- Bio/description length  
- Number of posts  
- Number of followers and following  
- Account privacy status  
- Presence of external URLs  

**Target Variable**  
- `fake` → Binary class (`Fake`, `Real`)

---

## 🔍 Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to:
- Understand feature distributions  
- Identify class imbalance  
- Detect outliers and anomalies  
- Examine relationships between predictors and the target variable  

Various visualizations such as histograms, bar plots, and boxplots were used to gain insights into user behavior patterns.

---

## 🛠️ Data Preprocessing & Feature Engineering

Data preprocessing ensured consistency and model readiness. Feature engineering focused on extracting behavioral and ratio-based indicators to enhance model performance.

### Engineered Features:
- Username numeric ratio  
- Fullname numeric ratio  
- Followers-to-following ratio  
- Posts-to-followers ratio  
- Binary indicators for:
  - Profile picture
  - External URL
  - Account privacy  

The engineered datasets were saved as CSV files for reproducibility.

---

## 🎯 Feature Selection

Feature selection was carried out using Random Forest variable importance. Less important features were removed to reduce noise and improve computational efficiency while maintaining predictive performance.

The final dataset consisted of **9 selected predictors**, used consistently across all models.

---

## 🤖 Model Building

The following machine learning models were implemented using the `caret` framework with cross-validation:

- Logistic Regression  
- Random Forest  
- XGBoost  
- Support Vector Machine (SVM – RBF Kernel)  

A unified resampling strategy was applied to ensure fair comparison across models.

---

## 📊 Model Evaluation

Models were evaluated using the following metrics:
- Receiver Operating Characteristic (ROC)
- Sensitivity
- Specificity  

### Cross-Validated Performance Summary

| Model | ROC | Sensitivity | Specificity |
|------|-----|------------|-------------|
| **Random Forest** | **0.9839** | 0.9302 | 0.9236 |
| XGBoost | 0.9822 | 0.9269 | **0.9272** |
| Logistic Regression | 0.9709 | 0.9302 | 0.9131 |
| SVM | 0.9609 | 0.9268 | 0.8815 |

Random Forest achieved the highest overall ROC and demonstrated stable performance across resamples.

---

## 🧠 Feature Importance Analysis

Feature importance analysis using Random Forest revealed that the most influential predictors were:
- Number of followers  
- Number of posts  
- Number of accounts followed  
- Profile picture presence  
- Username and fullname characteristics  

These results indicate that activity patterns and profile completeness play a critical role in identifying fake accounts.

---

## 🏆 Final Model Selection

Although XGBoost showed strong performance, particularly in specificity, Random Forest marginally outperformed all other models in terms of overall ROC and stability. Therefore, Random Forest was selected as the final model.

---

## 📈 Test Set Prediction

The final Random Forest model was applied to the unseen test dataset to generate predictions, enabling identification of potentially fake Instagram accounts for further analysis or moderation.

---

## 🧰 Technologies & Libraries Used

- **Language**: R  
- **Libraries**:
  - caret  
  - randomForest  
  - xgboost  
  - e1071  
  - dplyr  
  - ggplot2  
  - tidyverse  
  - pROC  

---
## 📝 Conclusion

This project demonstrates the effectiveness of machine learning techniques in detecting fake Instagram accounts. Ensemble-based methods, particularly Random Forest, proved to be the most reliable due to their ability to handle complex feature interactions and non-linear patterns.

---

## 🚀 Future Scope

- Incorporation of temporal activity-based features  
- Use of deep learning for profile image and bio text analysis  
- Extension to multiple social media platforms  
- Deployment as a real-time detection system  

---

## 👤 Author

**Abdul Razzaq**  
Master’s in Statistics | Aspiring Data Scientist  
