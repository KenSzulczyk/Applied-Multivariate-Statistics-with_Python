# Chapter 6-Multiple Regression-Case Study-Coffee Demand

# We now return to the coffee dataset introduced in Section 5.1. This dataset contains annual observations from 1990 to 2023 and includes several economically meaningful variables used to explain coffee consumption.
# -Coffee consumption (y): measured as cups per person per year
# 	-Coffee price (x₁): dollars per cup
# 	-Tea price (x₂): dollars per cup
# 	-Sugar price (x₃): dollars per pound
# 	-Annual income (x₄): dollars per household per year
# 

# In[1]:

# --------------------------------------------------
# Demand Estimation for Coffee using OLS Regression
# --------------------------------------------------

# Step 1: Import libraries
import pandas as pd                      # For data handling
import numpy as np                       # For mathematical functions (e.g., log)
import statsmodels.formula.api as smf    # For regression using formulas

# Step 2: Load the dataset from a CSV file
# Make sure "coffee_demand.csv" is in your working directory
df = pd.read_csv("Chapter_06-coffee_demand.csv")

# Step 3: Display the first few rows of the dataset
# This helps verify that the data loaded correctly
print(df.head())

# Step 4: Estimate the demand function using OLS
# The formula syntax allows us to write the model like an equation
# Coffee_consumption is the dependent variable (Q)
# The independent variables include prices and income
model = smf.ols(
    formula="Coffee_consumption ~ Coffee_price + Tea_price + Sugar_price + Annual_income",
    data=df).fit()

# Step 5: Print the regression results
# The summary includes coefficients, p-values, R-squared, and more
print(model.summary())

# The regression output in Section 6.2 reports an F-statistic, which comes directly from analysis of variance (ANOVA). In this section, we connect the classical ANOVA framework to multiple regression and show how it helps us understand model performance.
# 
# The key idea is simple: The total variation in the dependent variable can be divided into what the model explains and what it cannot explain.
# 
# We decompose the total variation in coffee consumption into three components:
# 
# SST=SSR+SSE

# In[3]:

# --------------------------------------------------
# Step 6: Classical ANOVA Table (Manual Construction)
# --------------------------------------------------

# Degrees of freedom
n = len(df)                # Number of observations
k = model.df_model         # Number of independent variables

df_reg = int(k)            # Regression degrees of freedom
df_err = int(n - k - 1)    # Error degrees of freedom
df_tot = int(n - 1)        # Total degrees of freedom

# Total Sum of Squares (SST)
# Measures total variation in the dependent variable
SST = df_tot * ( df["Coffee_consumption"].var() )

# Sum of Squared Errors (SSE)
# Measures unexplained variation (residuals)
SSE = sum(model.resid**2)

# Regression Sum of Squares (SSR)
# Measures explained variation
SSR = SST - SSE

# Mean Squares
MSR = SSR / df_reg         # Mean square regression
MSE = SSE / df_err         # Mean square error

# F-statistic
F = MSR / MSE

# Create a clean ANOVA table
anova_classic = pd.DataFrame({
    "Source": ["Regression", "Error", "Total"],
    "SS": [SSR, SSE, SST],
    "df": [df_reg, df_err, df_tot],
    "MS": [MSR, MSE, ""],
    "F": [F, "", ""]
})

print(anova_classic)

# We now extend our coffee demand model by including a dummy variable to capture the effect of the 2008–2009 financial crisis on coffee consumption.
# 
# D={ 1     if year equals 2008 or 2009
#     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0     otherwise }
# 
# This variable allows us to test whether coffee consumption behaved differently during the crisis years. We include the dummy (D) variable as an additional explanatory variable:
# 

# In[5]:

# Create dummy variable for financial crisis years
df["Crisis_dummy"] = ((df["Year"] >= 2008) & (df["Year"] <= 2009)).astype(int)

# Check the data
print(df[["Year", "Crisis_dummy"]].head(20))

# Estimate regression with dummy variable
model_dummy = smf.ols(
    formula="Coffee_consumption ~ Coffee_price + Tea_price + Sugar_price + Annual_income + Crisis_dummy",
    data=df
).fit()

print(model_dummy.summary())

# In regression analysis, we often face an important question: Should we include or exclude certain variables from the model? Earlier, we used adjusted R2 to evaluate whether adding a variable improves the model’s fit. However, adjusted R2 alone does not tell us whether the improvement is statistically significant. To formally test this, we use an F-test.
# 
# We begin with the demand model for coffee specified earlier in the chapter. This is called the full model because it includes all variables suggested by economic theory:
# 
# y = β0 + β1∙x1 + β2∙x2 + β3∙x3 + β4∙x4 + ε
# 
# where:
# 	x1: Coffee price
# 	x2: Tea price
# 	x3: Sugar price
# 	x4: Income
# 
# From the regression results, we observed that sugar price is not statistically significant, and we suspect that tea price may also contribute little to explaining coffee demand.
# 
# In practice, we often prefer a parsimonious model. It means a model that explains the data well using as few variables as possible. To test whether tea and sugar prices are necessary, we specify a reduced model that excludes these variables:
# 
# y = β0 + β1∙x1 + β4∙x4 + ε
# 
# We want to test whether tea and sugar prices are jointly significant. The hypotheses are below:
# 
# H0: β2 = β3 = 0
# Ha: At least one slope does not equal zero.
# 

# In[6]:

# --------------------------------------------------
# Demand Estimation for Coffee using OLS Regression
# --------------------------------------------------

# Step 1: Import libraries
import pandas as pd                      # For data handling
import numpy as np                       # For mathematical functions (e.g., log)
import statsmodels.formula.api as smf    # For regression using formulas

# Step 2: Load the dataset from a CSV file
# Make sure "coffee_demand.csv" is in your working directory
df = pd.read_csv("Chapter_06-coffee_demand.csv")

# Step 3: Estimate the demand function using OLS
# The formula syntax allows us to write the model like an equation
# Coffee_consumption is the dependent variable (Q)
# The independent variables include prices and income
model_full = smf.ols(
    formula="Coffee_consumption ~ Coffee_price + Tea_price + Sugar_price + Annual_income",
    data=df).fit()

# --------------------------------------------------
# Step 4: Test whether Tea_price and Sugar_price can be removed
# (Partial F-test using nested models)
# --------------------------------------------------

# Estimate the reduced model (omit Tea_price and Sugar_price)
model_reduced = smf.ols(
    formula="Coffee_consumption ~ Coffee_price + Annual_income",
    data=df).fit()

# Compare full model vs reduced model
# This performs the joint F-test:
# H0: β_Tea = β_Sugar = 0
f_test = model_full.compare_f_test(model_reduced)

# Display results
print("\nPartial F-test (Joint Significance Test):")
print("F-statistic:", f_test[0])
print("p-value:", f_test[1])
print("Degrees of freedom:", f_test[2])

# Residuals (e) are the estimated errors:
# 
# e = y - y_hat
# 
# They provide valuable information about whether the assumptions of the regression model are satisfied.
# 
# In simple linear regression, plotting residuals against x or against y ̂ provides similar information. However, in multiple regression, there are several explanatory variables, so it is more useful to plot residuals against the fitted values (y_hat).
# 
# Standardized residuals are commonly used to identify unusual observations. These residuals are scaled by their estimated standard deviation. They allow us to compare them across observations.

# In[7]:

# --------------------------------------------------
# Demand Estimation for Coffee using OLS Regression
# --------------------------------------------------

# Step 1: Import libraries
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Step 2: Load the dataset
df = pd.read_csv("Chapter_06-coffee_demand.csv")

# Step 3: Display the first few rows
print(df.head())

# Step 4: Estimate the demand function using OLS
model = smf.ols(
    formula="Coffee_consumption ~ Coffee_price + Tea_price + Sugar_price + Annual_income",
    data=df).fit()

# Step 5: Print regression results
print(model.summary())

# --------------------------------------------------
# Step 6: Extract residuals
# --------------------------------------------------

# Regular residuals
df["residuals"] = model.resid

# Standardized residuals
influence = model.get_influence()
df["standardized_residuals"] = influence.resid_studentized_internal

# Display first few values
print("\nResiduals:")
print(df[["residuals", "standardized_residuals"]].head())

# --------------------------------------------------
# Step 7: Plot Regular Residuals
# --------------------------------------------------

plt.figure()

# Scatter plot: fitted values vs residuals
plt.scatter(df["Year"], df["residuals"])

# Horizontal reference line at zero
plt.axhline(y=0, color="darkred")

# Labels and title
plt.xlabel("Year")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")

plt.show()

# --------------------------------------------------
# Step 8: Plot Standardized Residuals
# --------------------------------------------------

plt.figure()

# Scatter plot: fitted values vs standardized residuals
plt.scatter(df['Year'], df["standardized_residuals"])

# Reference lines
plt.axhline(y=0, color="cornflowerblue")
plt.axhline(y=2, color="darkred")
plt.axhline(y=-2, color="darkred")

# Labels and title
plt.xlabel("Year")
plt.ylabel("Standardized Residuals")
plt.title("Standardized Residals vs Fitted Values")

plt.show()

# In[ ]:
