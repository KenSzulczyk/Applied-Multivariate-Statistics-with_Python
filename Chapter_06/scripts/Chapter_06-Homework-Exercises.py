# Chapter 6-Homework-Exercises

# Problem 1.
# 
# We want to estimate how different factors affect house prices. We collect data on:
# 
# Price (in $1000s) dependent variable
# 	Size (square meters)
# 	Bedrooms (number of bedrooms)
# 	Age (years since construction)
# 
# Please estimate the regression below and evaluate the results:
# 
# Price = β0 + β1∙Size + β2∙Bedrooms + β3∙Age + ε
# 

# In[2]:

# --------------------------------------------------
# Multiple Regression Example: Housing Prices
# --------------------------------------------------

# Import libraries
import pandas as pd
import statsmodels.api as sm

# Create dataset manually
data = {
    "Size": [50,60,80,100,120,140,160,180,70,90,110,130,150,170,200],
    "Bedrooms": [1,2,3,3,4,4,5,5,2,3,3,4,4,5,6],
    "Age": [20,15,10,5,8,3,2,1,12,9,7,6,4,2,1],
    "Price": [120,150,200,250,280,320,360,400,170,220,260,300,340,380,420]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Display dataset
print(df)

# Define independent variables
X = df[["Size", "Bedrooms", "Age"]]
X = sm.add_constant(X)

# Define dependent variable
y = df["Price"]

# Estimate regression model
model = sm.OLS(y, X).fit()

# Print results
print(model.summary())

# Problem 2.
# 
# Write a Python program to complete the following tasks:
# 
# a. Import the dataset Chapter_06-cubic_cost_function.csv into Python and display the first few observations.
# 
# b. Create a scatter plot showing the relationship between total cost (TC) and quantity produced (q). Briefly describe the pattern we observe.
# 
# c. Generate two additional variables:
# 	q2 (quantity squared)
# 	q3 (quantity cubed)
# 
# Then compute and display the correlation matrix for q, q2, q3, and TC.
# 
# d. Estimate the following cubic cost function using ordinary least squares (OLS):
# 
# TC = β0 + β1∙q + β2∙q^2 + β3∙q^3 + ε
# 
# e. Interpret the regression results. In our answer, comment on:
# 	The statistical significance of the coefficients
# 	The overall fit of the model
# Whether the estimated relationship is consistent with a typical total cost function
# 

# In[3]:

# --------------------------------------------------
# Cubic Cost Function with Noise
# Data Import, Visualization, and Estimation
# --------------------------------------------------

# Step 1: Import libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# --------------------------------------------------
# Step 2: Load the dataset
# --------------------------------------------------

# Make sure the CSV file is in your working directory
df = pd.read_csv("Chapter_06-cubic_cost_function.csv")

# Display first few rows
print("Dataset:")
print(df.head())

# --------------------------------------------------
# Step 3: Plot Total Cost vs Quantity
# --------------------------------------------------

plt.figure()

# Scatter plot
plt.scatter(df["Quantity"], df["Total_Cost"])

# Labels and title
plt.xlabel("Quantity")
plt.ylabel("Total Cost")
plt.title("Total Cost vs Quantity")

plt.show()

# --------------------------------------------------
# Step 4: Create squared and cubic terms
# --------------------------------------------------

df["Quantity_sq"] = df["Quantity"]**2
df["Quantity_cu"] = df["Quantity"]**3

print("\nDataset with Polynomial Terms:")
print(df.head())

# --------------------------------------------------
# Step 5: Correlation Matrix
# --------------------------------------------------

corr_matrix = df[["Quantity", "Quantity_sq", "Quantity_cu", "Total_Cost"]].corr()

print("\nCorrelation Matrix:")
print(corr_matrix)

# --------------------------------------------------
# Step 6: Estimate Cubic Regression Model
# --------------------------------------------------

# Define independent variables
X = df[["Quantity", "Quantity_sq", "Quantity_cu"]]
X = sm.add_constant(X)

# Dependent variable
y = df["Total_Cost"]

# Estimate model
model = sm.OLS(y, X).fit()

# Print regression results
print("\nRegression Results:")
print(model.summary())

# In[ ]: