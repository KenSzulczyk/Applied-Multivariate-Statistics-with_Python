# Chapter 7-Homework-Problem 2

# In this exercise, we analyze a dataset on teenage gambling behavior using both regression and ANOVA techniques. The dataset was originally obtained from an R library.
# 
# The data frame contains the following variables:
# 
# Dependent Variable
# •	Gamble: Annual expenditure on gambling (measured in pounds per year)
# 
# Categorical Variable (Factor): Sex:
# •	0 = Male
# •	1 = Female
# 
# Covariates (Continuous Variables)
# •	Status: Socioeconomic status score based on parents’ occupation
# •	Income: Weekly income (in pounds)
# •	Verbal: Verbal ability score (number of words correctly defined out of 12)
# 
# a. Estimate a multiple regression model using Gamble as the dependent variable and Sex, Status, Income, and Verbal as explanatory variables. Interpret the results.
# 
# b. Perform an ANOVA-style analysis by treating Sex as the factor and Status, Income, and Verbal as covariates, i.e., estimate an ANCOVA model. Compare the results with those from part (a).
# 

# In[1]:

# ============================================
# ANCOVA Example: Teen Gambling Dataset
# ============================================

# --------------------------------------------
# Step 1: Import required libraries
# --------------------------------------------
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# --------------------------------------------
# Step 2: Load the dataset
# --------------------------------------------
teengamb = pd.read_csv('Chapter_07-teen_gambling.csv')

print("First five observations:")
print(teengamb.head())

# --------------------------------------------
# Step 3: Estimate regression model (OLS)
# --------------------------------------------
# This step helps us understand relationships
# before performing ANCOVA

model = ols(
    'gamble ~ income + sex + status + verbal',
    data=teengamb
).fit()

print("\nOLS Regression Results:")
print(model.summary())

# --------------------------------------------
# Step 4: Perform ANCOVA
# --------------------------------------------
# dv = dependent variable (gamble)
# covar = continuous control variables
# between = categorical factor (sex)

from pingouin import ancova

ancova_results = ancova(
    data=teengamb,
    dv='gamble',
    covar=['income', 'status', 'verbal'],
    between='sex'
)

print("\nANCOVA Results:")
print(ancova_results)

# In[ ]:




