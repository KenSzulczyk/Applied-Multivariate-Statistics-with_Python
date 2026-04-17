# ============================================
# MANOVA Example: Training Methods Experiment
# ============================================

# Import required libraries
import pandas as pd
import numpy as np
from statsmodels.multivariate.manova import MANOVA

# --------------------------------------------
# Step 1: Load the dataset
# --------------------------------------------
# Make sure the file path is correct
df = pd.read_excel('Chapter_07-MANOVA.xlsx')

# Display first few rows to verify data
print(df.head())

# --------------------------------------------
# Step 2: Create transformed variable
# --------------------------------------------
# Take the natural logarithm of 'Correct'
# Add a small constant if needed to avoid log(0)
df['ln_Correct'] = np.log(df['Correct'])

# --------------------------------------------
# Step 3: MANOVA without transformation
# --------------------------------------------
# Model includes:
# - Main effects: Method, Sessions
# - Interaction: Method * Sessions
model_original = MANOVA.from_formula(
    'Correct + Delay ~ Method + Sessions + Method:Sessions',
    data=df
)

print("\nMANOVA Results (Original Variables):")
print(model_original.mv_test())

# --------------------------------------------
# Step 4: MANOVA with transformation
# --------------------------------------------
model_transformed = MANOVA.from_formula(
    'ln_Correct + Delay ~ Method + Sessions + Method:Sessions',
    data=df
)

print("\nMANOVA Results (Log-Transformed Correct):")
print(model_transformed.mv_test())
