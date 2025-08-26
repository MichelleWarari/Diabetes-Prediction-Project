# Diabetes Prediction Project

## 1 Introduction

This project aims to predict diabetes outcomes using a dataset with eight independent features (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age) and one dependent variable (Outcome). The analysis leverages machine learning techniques, including Logistic Regression (LR), Decision Tree (DC), Support Vector Machine (SVM), and K-Nearest Neighbors (KNN), to evaluate their performance in predicting diabetes.

## 2 Dataset Analysis

The dataset comprises 768 entries with nine columns, including eight numerical features and the binary Outcome variable (0 for non-diabetic, 1 for diabetic). Initial exploration confirms no missing values, with the dataset split into 500 non-diabetic and 268 diabetic cases. Statistical summaries reveal key insights, such as mean Glucose levels of 120.89 and a standard deviation of 31.97, indicating variability in the data. A count plot visualizes the class distribution, highlighting an imbalance between non-diabetic and diabetic cases.

### 2.1 Code Implementation for Data Loading and Exploration

#### 2.1.1 Importing Libraries

```python
# Importing multiple libraries to read analyze, and visualize the dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

#### 2.1.2 Loading the Dataset

```python
# Loading the diabetes dataset
dataset = pd.read_csv("/content/diabetes.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

#### 2.1.3 Displaying Dataset Head

```python
# Displaying the dataset
dataset.head()
```

Sample Output:

| Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin | BMI | DiabetesPedigreeFunction | Age | Outcome |
|-------------|---------|---------------|---------------|---------|-----|--------------------------|-----|---------|
| 6           | 148     | 72            | 35            | 0       | 33.6 | 0.627                   | 50  | 1       |
| 1           | 85      | 66            | 29            | 0       | 26.6 | 0.351                   | 31  | 0       |
| 8           | 183     | 64            | 0             | 0       | 23.3 | 0.672                   | 32  | 1       |
| 1           | 89      | 66            | 23            | 94      | 28.1 | 0.167                   | 21  | 0       |
| 0           | 137     | 40            | 35            | 168     | 43.1 | 2.288                   | 33  | 1       |

#### 2.1.4 Checking Dataset Shape

```python
# Checking the dataset shape
dataset.shape
```

Output: `(768, 9)`

#### 2.1.5 Dataset Information

```python
# Basic information regarding data
dataset.info()
```

Sample Output:
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 768 entries, 0 to 767
Data columns (total 9 columns):
 #   Column                    Non-Null Count  Dtype  
---  ------                    --------------  -----  
 0   Pregnancies               768 non-null    int64  
 1   Glucose                   768 non-null    int64  
 2   BloodPressure             768 non-null    int64  
 3   SkinThickness             768 non-null    int64  
 4   Insulin                   768 non-null    int64  
 5   BMI                       768 non-null    float64
 6   DiabetesPedigreeFunction  768 non-null    float64
 7   Age                       768 non-null    int64  
 8   Outcome                   768 non-null    int64  
dtypes: float64(2), int64(7)
memory usage: 54.1 KB
```

#### 2.1.6 Checking for Missing Values

```python
# Finding the missing values
dataset.isnull().sum()
```

Output:
```
Pregnancies                 0
Glucose                     0
BloodPressure               0
SkinThickness               0
Insulin                     0
BMI                         0
DiabetesPedigreeFunction    0
Age                         0
Outcome                     0
dtype: int64
```

#### 2.1.7 Statistical Summary

```python
# Describe function gives the basic numerical info about data for each numeric feature
print(dataset.describe(include='all'))
```

Sample Output:

|       | Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin | BMI       | DiabetesPedigreeFunction | Age     | Outcome |
|-------|-------------|---------|---------------|---------------|---------|-----------|--------------------------|---------|---------|
| count | 768.000000  | 768.000000 | 768.000000 | 768.000000 | 768.000000 | 768.000000 | 768.000000             | 768.000000 | 768.000000 |
| mean  | 3.845052    | 120.894531 | 69.105469    | 20.536458    | 79.799479 | 31.992578 | 0.471876               | 33.240885 | 0.348958 |
| std   | 3.369578    | 31.972618  | 19.355807    | 15.952218    | 115.244002 | 7.884160  | 0.331329               | 11.760232 | 0.476951 |
| min   | 0.000000    | 0.000000   | 0.000000     | 0.000000     | 0.000000   | 0.000000  | 0.078000               | 21.000000 | 0.000000 |
| 25%   | 1.000000    | 99.000000  | 62.000000    | 0.000000     | 0.000000   | 27.300000 | 0.243750               | 24.000000 | 0.000000 |
| 50%   | 3.000000    | 117.000000 | 72.000000    | 23.000000    | 30.500000  | 32.000000 | 0.372500               | 29.000000 | 0.000000 |
| 75%   | 6.000000    | 140.250000 | 80.000000    | 32.000000    | 127.250000 | 36.600000 | 0.626250               | 41.000000 | 1.000000 |
| max   | 17.000000   | 199.000000 | 122.000000   | 99.000000    | 846.000000 | 67.100000 | 2.420000               | 81.000000 | 1.000000 |

#### 2.1.8 Class Distribution

```python
# Data points count value for each class labels
dataset.Outcome.value_counts()
```

Output:
```
Outcome
0    500
1    268
Name: count, dtype: int64
```

#### 2.1.9 Visualizing Class Distribution

```python
# Visualize the distribution of Outcome (target variable)
sns.countplot(x='Outcome', data=dataset)
plt.show()
```

Output: A bar plot with two bars: one for Outcome 0 (non-diabetic, 500) and one for Outcome 1 (diabetic, 268).

## 3 Model Performance

The performance of the four machine learning models--Logistic Regression (LR), Decision Tree (DC), Support Vector Machine (SVM), and K-Nearest Neighbors (KNN)--is evaluated using sensitivity and specificity metrics. Sensitivity measures the models' ability to correctly identify diabetic patients, while specificity assesses their accuracy in identifying non-diabetic individuals. The results indicate that the Decision Tree model achieves the highest sensitivity, making it the most effective at detecting diabetic cases. Conversely, Logistic Regression demonstrates the highest specificity, excelling at identifying non-diabetic individuals, followed by SVM, KNN, and Decision Tree.

### 3.1 Code for Model Performance

The following Python code implements the training and evaluation of the four machine learning models. Sensitivity and specificity are calculated based on the confusion matrix for each model.

```python
# Importing necessary libraries for model training and evaluation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing the models
lr = LogisticRegression(max_iter=1000, random_state=42)
dt = DecisionTreeClassifier(random_state=42)
svm = SVC(random_state=42)
knn = KNeighborsClassifier()

# Training the models
lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
svm.fit(X_train, y_train)
knn.fit(X_train, y_train)

# Function to calculate sensitivity and specificity
def calc_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity

# Evaluating the models
models = {'Logistic Regression': lr, 'Decision Tree': dt, 'SVM': svm, 'KNN': knn}
results = {}

for name, model in models.items():
    y_pred = model.predict(X_test)
    sensitivity, specificity = calc_metrics(y_test, y_pred)
    results[name] = {'Sensitivity': sensitivity, 'Specificity': specificity}

# Displaying results
for name, metrics in results.items():
    print(f"{name}: Sensitivity = {metrics['Sensitivity']:.3f}, Specificity = {metrics['Specificity']:.3f}")
```

Input: Features (X) and target (y) from the dataset, split into 80% training and 20% testing sets.
Output: Sensitivity and specificity metrics for each model (Logistic Regression, Decision Tree, SVM, KNN).
Sample Output: (Note: Actual values depend on the dataset and random seed.)

```
Logistic Regression: Sensitivity = 0.611, Specificity = 0.827
Decision Tree: Sensitivity = 0.759, Specificity = 0.692
SVM: Sensitivity = 0.574, Specificity = 0.808
KNN: Sensitivity = 0.630, Specificity = 0.769
```

## 4 Conclusion

The Diabetes Prediction Project demonstrates the application of machine learning models to predict diabetes outcomes. The Decision Tree model excels in identifying diabetic patients (highest sensitivity), while Logistic Regression is the most effective at identifying non-diabetic individuals (highest specificity). These findings highlight the trade-offs between sensitivity and specificity among the models, with Decision Tree and Logistic Regression offering complementary strengths. Future work could explore techniques like dataset balancing or ensemble methods to further enhance predictive performance.
