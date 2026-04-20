# Chapter 5-Linear Regression

# We consider a dataset that relates the number of hours a student studies to their exam score. Our goal is to estimate whether study time has a measurable effect on performance. We begin by creating the dataset and estimating a simple linear regression model using the statsmodels library.

# In[1]:

# Import libraries
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Step 1: Create the dataset
data = {
    "Hours": [2, 4, 6, 8, 10],
    "Score": [65, 67, 82, 85, 84]
}

# Create a Pandas dataframe
df = pd.DataFrame(data)

# Step 2: (Already in correct format)
df_long = df

# Step 3: Fit the regression model
model = ols('Score ~ Hours', data=df_long).fit()

# Step 4: Get ANOVA table
anova_table = sm.stats.anova_lm(model, typ=1)

# Step 5: Print results
print("\n=== ANOVA Table (Regression) ===")
print(anova_table)

print("\n=== Regression Summary ===")
print(model.summary())

# In[ ]:




