# Multivariate Analysis of Variance (MANOVA)

# To illustrate MANOVA, consider an experiment that evaluates different training methods for call center recruits handling travel-related customer queries.
# 
# We examine two types of training methods:
# 
# •	Method 1: Trainees spend part of the first session learning the transportation network using maps and printed timetables available to the public.
# •	Method 2: Trainees immediately begin using the online information system that they will rely on when responding to customer queries.
# 
# In addition to training method, we vary the number of training sessions: recruits receive either one, two, or three sessions. Combining these factors gives us a 2 × 3 factorial design, resulting in six experimental conditions.
# 
# We randomly assign 30 new recruits to these conditions, with five participants in each group. Random assignment helps ensure that any differences in performance are due to the training methods rather than pre-existing differences among recruits.
# 

# In[1]:

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

# In[ ]:
