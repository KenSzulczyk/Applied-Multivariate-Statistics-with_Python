# Chapter 7-Logistic Regression

# We apply logistic regression to study whether individuals develop diabetes. The dependent variable is:
# 
# •	y = 1: has diabetes
# •	y = 0: does not have diabetes
# 
# The explanatory variables include:
# 
# •	Pregnant – number of pregnancies
# •	Insulin – insulin level
# •	BMI – body mass index
# •	Age – age of the individual
# •	Glucose – blood glucose level
# •	BP – blood pressure
# •	Pedigree – genetic predisposition
# 

# In[1]:

# ============================================
# Logistic Regression Example: Diabetes Dataset
# ============================================

# --------------------------------------------
# Step 1: Import required libraries
# --------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report

import statsmodels.api as sm

# --------------------------------------------
# Step 2: Load and clean the dataset
# --------------------------------------------

# Load dataset
df = pd.read_excel('Chapter_07-diabetes.xlsx')

# Convert all columns to numeric (important for modeling)
df = df.apply(pd.to_numeric)

# Preview data
print("First five observations:")
print(df.head())

# --------------------------------------------
# Step 3: Define features (X) and target (y)
# --------------------------------------------
feature_cols = [
    'pregnant', 'insulin', 'bmi',
    'age', 'glucose', 'bp', 'pedigree'
]

X = df[feature_cols]   # Independent variables
y = df['outcome']      # Dependent variable

# --------------------------------------------
# Step 4: Split into training and testing sets
# --------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=16
)

# --------------------------------------------
# Step 5: Estimate logistic regression (sklearn)
# --------------------------------------------
model = LogisticRegression(max_iter=1000)

# Fit the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# --------------------------------------------
# Step 6: Evaluate model performance
# --------------------------------------------
# Confusion matrix
conf_matrix = metrics.confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(conf_matrix)

# --------------------------------------------
# Step 7: Visualize confusion matrix
# --------------------------------------------
plt.figure()
sns.heatmap(
    pd.DataFrame(conf_matrix),
    annot=True,
    fmt='g',
    cmap = 'Blues'
)

plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# --------------------------------------------
# Step 8: Classification report
# --------------------------------------------
target_names = ['Without Diabetes', 'With Diabetes']

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# --------------------------------------------
# Step 9: Estimate Logit model (statsmodels)
# --------------------------------------------
# Add constant for intercept
X_train_sm = sm.add_constant(X_train)

# Fit Logit model
logit_model = sm.Logit(
    y_train.astype(float),
    X_train_sm.astype(float)
).fit()

# Display results
print("\nLogit Model Summary:")
print(logit_model.summary())

# In[ ]:

