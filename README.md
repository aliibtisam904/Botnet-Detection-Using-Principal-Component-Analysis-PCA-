# Botnet Detection Using Principal Component Analysis

## Project Overview

This project focuses on the detection of botnet activity in network traffic using Principal Component Analysis (PCA) for dimensionality reduction and various machine learning algorithms for classification. 

## Team Members

- Ali Ibtisam (23I-2009)
- Muhammad Hussain (23I-2070)
- Khawaja Fashi Ud Din Abdullah (23I-2077)
- Lutfan Shahzad (23I-2114)
- Musab Ahmed (23I-2132)

Supervised by: Mr. Ahtasham-ul-Haq

## Institution

Department of Cyber Security, National University of Computer and Emerging Sciences, Islamabad, Pakistan

## Project Details

### 1. Data Collection

- Collection of network traffic data.
- Extraction of features such as packet size, protocol type, source and destination IP addresses, port numbers, etc.

### 2. Feature Extraction

- Conversion of raw data into a suitable format for analysis.
- Extraction of pertinent features from network packets and transformation into numerical representation.

### 3. Normalization

- Scaling features to a consistent range to ensure fair representation across all dimensions.

### 4. PCA Application

- Identification of principal components that capture the maximum variance in the data.
- Use of orthogonal principal components to represent independent aspects of the data.

### 5. Dimensionality Reduction

- Selection of a subset of principal components that explain the majority of the variance.
- Reduction of data dimensionality to facilitate easier analysis and interpretation.

### 6. Model Training and Testing

- Training of machine learning models (e.g., Support Vector Machines, Random Forest, Neural Networks) using reduced-dimensional data.
- Evaluation of models on separate test datasets to gauge performance.

### 7. Evaluation

- Assessment of model effectiveness using metrics such as accuracy, precision, recall, and F1-score.

### 8. Deployment

- Deployment of the botnet detection model in real-world network environments.
- Continuous monitoring and detection of botnet activity to enhance network security.

## Weekly Progress

### Week 1: Introduction to PCA and Dimensionality Reduction Techniques

- Understanding of PCA and other methods for reducing data complexity.

### Week 2: Applying PCA to Reduce Dimensionality and Developing Botnet Detection Algorithms

- Creation and loading of synthetic network traffic data.
- Data preprocessing, including separation into features and labels, encoding categorical variables, and standardizing features.
- Application of PCA to perform dimensionality reduction.
- Training and evaluation of a logistic regression model to classify network traffic.

## Code and Visualization

### Importing Libraries

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
```

### PCA Scatter Plot

```python
def visualize_pca(X_train_pca, y_train):
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    plt.figure(figsize=(10, 8))
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train_encoded)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA Scatter Plot')
    plt.show()
```

### Confusion Matrix Visualization

```python
def visualize_confusion_matrix(y_test, y_pred):
    conf_mat = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
```

## Challenges and Solutions

### Problems Faced

- Unfamiliarity with the topics and Python programming language.

### Solutions

- Utilization of online resources and extensive research.
- Team collaboration and guidance from the team lead.

## Learning Outcomes

- Enhanced understanding of team work and Python programming.
- Knowledge of PCA and its application in botnet detection.

## Individual Contributions

- **Team Lead (Kh. Fashi)**: Understanding botnet functionality and leading the coding efforts.
- **M. Hussain**: Assisting in coding.
- **Other Team Members**: Contributing to coding and report writing.

## References

- "Data Mining: Concepts and Techniques" by Jiawei Han, Micheline Kamber, and Jian Pei.
- Scikit-learn documentation for implementing PCA and building machine learning models in Python.
