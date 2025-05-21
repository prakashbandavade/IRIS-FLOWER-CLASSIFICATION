
# 🌸 Iris Flower Classification

This project performs classification on the **Iris Flower dataset** using machine learning models. The goal is to predict the species of an Iris flower (Setosa, Versicolor, or Virginica) based on its sepal and petal dimensions.

---

## 📁 Dataset

The dataset used is the classic **Iris dataset**, originally introduced by Ronald A. Fisher.
It contains **150 records** and **4 features**:

* Sepal Length (cm)
* Sepal Width (cm)
* Petal Length (cm)
* Petal Width (cm)
* Target variable: `Species` (Setosa, Versicolor, Virginica)

> 📂 File used: `IRIS (1).csv`

---

## 📚 Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

---

## 🧠 Models Used

* Random Forest Classifier
* (Optional) Support Vector Machine, K-Nearest Neighbors, Logistic Regression (for comparison)

---

## 🔍 Exploratory Data Analysis (EDA)

* Pair plots and histograms for feature distribution
* Heatmap for correlation analysis
* Box plots for outlier detection

---

## ⚙️ Workflow

1. **Data Preprocessing**

   * Handle missing values (if any)
   * Encode categorical labels

2. **Train-Test Split**

   * 80% training, 20% testing

3. **Model Training**

   * Train a Random Forest Classifier

4. **Evaluation**

   * Accuracy Score
   * Confusion Matrix
   * Classification Report

---

## 📊 Results

Achieved an accuracy of **\~95–100%** (depending on the random state and model used) using the Random Forest Classifier.

---

## 📌 How to Run

1. Clone the repo or download the notebook
2. Make sure `IRIS (1).csv` is in the same directory
3. Run the notebook step-by-step using Jupyter or Google Colab

---

## 📎 Project Structure

```
Iris-Flower-Classification/
│
├── IRIS (1).csv
├── Iris_Classification.ipynb
└── README.md
```

---

## 🏷️ Tags

`#MachineLearning` `#IrisDataset` `#Classification` `#RandomForest` `#DataScience`

