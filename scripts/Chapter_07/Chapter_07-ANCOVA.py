# ============================================
# ANCOVA Example: Study Techniques
# ============================================

# --------------------------------------------
# Step 1: Import required libraries
# --------------------------------------------
import numpy as np
import pandas as pd

# --------------------------------------------
# Step 2: Create the dataset
# --------------------------------------------
# We simulate data for three study techniques (A, B, C)
# Each student has:
# - current_grade (covariate)
# - exam_score (dependent variable)

df = pd.DataFrame({
    'technique': np.repeat(['A', 'B', 'C'], 5),
    'current_grade': [
        67, 88, 75, 77, 85,
        92, 69, 77, 74, 88,
        96, 91, 88, 82, 80
    ],
    'exam_score': [
        77, 89, 72, 74, 69,
        78, 88, 93, 94, 90,
        85, 81, 83, 88, 79
    ]
})

# View the dataset
print("Dataset:")
print(df)

# --------------------------------------------
# Step 3: Install and import ANCOVA library
# --------------------------------------------
# Run this line once in your terminal or notebook
# pip install pingouin

from pingouin import ancova

# --------------------------------------------
# Step 4: Perform ANCOVA
# --------------------------------------------
# dv = dependent variable (exam_score)
# covar = covariate (current_grade)
# between = factor (technique)

results = ancova(
    data=df,
    dv='exam_score',
    covar='current_grade',
    between='technique'
)

# Display results
print("\nANCOVA Results:")
print(results)
