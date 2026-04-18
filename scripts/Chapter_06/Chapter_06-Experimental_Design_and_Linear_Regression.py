# Chapter 6-Experimental Design and Linear Regression

# Lazer Manufacturing wants to determine whether managers at its three plants work different numbers of hours per week. The plants are located in Chicago, St. Louis, and Detroit. The data are summarized in Table 6.1.

# In[1]:

# -----------------------------------------------
# Regression with Dummy Variables (Experimental Design)
# -----------------------------------------------

# Import required libraries
import pandas as pd
import statsmodels.api as sm

# -----------------------------------------------
# STEP 1: Create the dataset directly in Python
# (No CSV needed since the dataset is small)
# -----------------------------------------------

data = {
    # Observation number (just for reference)
    "Obs": list(range(1, 16)),
    
    # Categorical variable: City
    "City": ["Chicago"]*5 + ["St. Louis"]*5 + ["Detroit"]*5,
    
    # Dummy variable A:
    # A = 1 if St. Louis, 0 otherwise
    "A": [0,0,0,0,0, 1,1,1,1,1, 0,0,0,0,0],
    
    # Dummy variable B:
    # B = 1 if Detroit, 0 otherwise
    "B": [0,0,0,0,0, 0,0,0,0,0, 1,1,1,1,1],
    
    # Dependent variable (response)
    "y": [49,53,58,62,53, 74,62,67,74,63, 50,64,60,56,55]
}

# Convert dictionary into a pandas DataFrame
df = pd.DataFrame(data)

# Display the dataset
print("Dataset:")
print(df)

# -----------------------------------------------
# STEP 2: Calculate the mean for each city
# (This is equivalent to group means in ANOVA)
# -----------------------------------------------

city_means = df.groupby("City")["y"].mean()

print("\nCity Means:")
print(city_means)

# Expected results:
# Chicago ≈ 55
# St. Louis ≈ 68
# Detroit ≈ 57

# -----------------------------------------------
# STEP 3: Set up regression model
# y = β0 + β1*A + β2*B + ε
# -----------------------------------------------

# Independent variables (dummy variables)
X = df[["A", "B"]]

# Add a constant (intercept term β0)
# This represents the baseline group (Chicago)
X = sm.add_constant(X)

# Dependent variable
y = df["y"]

# -----------------------------------------------
# STEP 4: Estimate the regression model and display results
# -----------------------------------------------

model = sm.OLS(y, X).fit()

print("\nRegression Results:")
print(model.summary())

# In[ ]:

