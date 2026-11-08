
# Breast Cancer Classification using Machine Learning

## 1. Problem Statement

The objective of this project is to develop and compare five Machine Learning classification algorithms to classify breast cancer samples as Malignant or Benign.

The project also provides an interactive Streamlit application for evaluating the trained models using uploaded test datasets.

## 2. Dataset Description

Dataset Name: Breast Cancer Wisconsin (Diagnostic)

Dataset Source: UCI Machine Learning Repository

Dataset Link:
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

- Total Instances: 569
- Total Input Features: 30
- Target Variable: target
- Classification Type: Binary Classification
- Target Classes: Malignant (0), Benign (1)
- Missing Values: 0

The dataset contains numerical measurements of cell nuclei obtained from digitized images of breast mass samples.

## 3. GitHub Repository Link

Add your GitHub repository URL here after creating the repository.

## 4. Machine Learning Models Used

Five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

### Data Preprocessing

- Dataset divided into 80% training and 20% testing.
- Stratified train-test splitting was used.
- StandardScaler was applied to standardize numerical features.
- Random state was set to 42.

### Model Performance Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

### Model Performance Observations

| Model | Observation |
|---|---|
| Logistic Regression | Achieved 98.25% accuracy and the highest AUC score of 0.9954. It demonstrated excellent classification performance. |
| Decision Tree | Achieved 91.23% accuracy. Its performance was comparatively lower than the other models. |
| KNN | Achieved 95.61% accuracy with good recall and F1 score. |
| Naive Bayes | Achieved 92.98% accuracy and a strong AUC score of 0.9868. |
| Support Vector Machine | Achieved 98.25% accuracy and excellent overall classification performance. |

### Overall Winner

Logistic Regression is the overall winner based on its highest AUC score of 0.9954, while achieving 98.25% accuracy.

Support Vector Machine also achieved 98.25% accuracy and performed very competitively.

## 5. Streamlit Application

The Streamlit application provides:

- CSV dataset upload functionality.
- Machine Learning model selection.
- Six evaluation metrics.
- Confusion matrix visualization.
- Classification report.
- Prediction results and CSV download.

The application loads previously trained models and evaluates uploaded test data without retraining.

## 6. Project Structure

```text
ML_Assignment_2/
│
├── app.py
├── train.py
├── ML_assignment_2.ipynb
├── requirements.txt
├── README.md
├── test_data.csv
│
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── support_vector_machine.pkl
    └── scaler.pkl
```

## 7. Conclusion

Five classification algorithms were successfully implemented and evaluated on the Breast Cancer Wisconsin dataset.

Logistic Regression and Support Vector Machine achieved the highest accuracy of 98.25%.

The project demonstrates the complete Machine Learning workflow, including dataset preprocessing, model training, performance evaluation, model saving, and interactive web application deployment.
