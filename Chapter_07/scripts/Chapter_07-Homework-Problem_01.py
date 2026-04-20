# Chapter 7-Homework-Problem 1

# In this exercise, we will apply Multivariate Analysis of Variance (MANOVA) using the well-known Iris dataset. This dataset, originally introduced by Ronald Fisher in 1936, is widely used for statistical analysis, machine learning, and data visualization.
# 
# The dataset contains measurements on iris flowers from three different species. For each observation, the following four variables are recorded:
# 
# •	Sepal length
# •	Sepal width
# •	Petal length
# •	Petal width
# 
# These four variables will serve as the dependent variables in your analysis. The independent variable is the species of the iris flower, which classifies each observation into one of three groups.
# 
# The task is to estimate a MANOVA using the four dependent variables and species as the independent variable. Then we test whether there are significant differences among species using the multivariate test statistics, e.g., Pillai’s Trace, Wilks’ Lambda.
# 

# In[1]:

# ============================================
# MANOVA Example: Iris Dataset
# ============================================

# --------------------------------------------
# Step 1: Import required libraries
# --------------------------------------------
import pandas as pd
from statsmodels.multivariate.manova import MANOVA
from sklearn.datasets import load_iris

# --------------------------------------------
# Step 2: Load the dataset
# --------------------------------------------
# Load iris dataset from sklearn
data = load_iris()

# Convert to pandas DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add species (target variable)
df['species'] = data.target

# Preview the data
print("First five observations:")
print(df.head())

# --------------------------------------------
# Step 3: Clean and rename variables
# --------------------------------------------
# Rename columns for easier use in formulas
df.columns = [
    'sepal_length',
    'sepal_width',
    'petal_length',
    'petal_width',
    'species'
]

# --------------------------------------------
# Step 4: Convert numeric labels to names
# --------------------------------------------
# Replace numeric species codes with labels
df['species'] = df['species'].map({
    0: 'setosa',
    1: 'versicolor',
    2: 'virginica'
})

# Verify changes
print("\nUnique species categories:")
print(df['species'].unique())

# --------------------------------------------
# Step 5: Perform MANOVA
# --------------------------------------------
# Define MANOVA model:
# Dependent variables: four flower measurements
# Independent variable: species
model = MANOVA.from_formula(
    'sepal_length + sepal_width + petal_length + petal_width ~ species',
    data=df
)

# Fit model and display results
print("\nMANOVA Results:")
print(model.mv_test())

# In[ ]:
