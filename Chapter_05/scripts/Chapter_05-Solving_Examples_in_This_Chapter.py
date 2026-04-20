# Chapter 5-Solve Python Examples in This Chapter

# Before estimating any formal econometric model, it is essential to explore and understand the data. Descriptive statistics and correlation analysis provide a first look at the structure of the dataset, helping us identify patterns, potential anomalies, and relationships between variables.
# 
# For now, we manually enter the dataset directly into Python. This approach allows students to focus on data structure and analysis without introducing file handling. In Chapter 6, we will extend this approach by importing datasets from CSV and Excel files.
# 

# In[1]:

# Import libraries
import pandas as pd

# Step 1: Create the dataset
data = {
    "Year": [
        1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,
        2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,
        2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,
        2020,2021,2022,2023
    ],
    "Coffee_consumption": [
        267.22,270.57,309.06,301.31,214.79,227.25,258.69,197.95,256.66,292.83,
        312.61,362.14,369.48,378.24,358.46,335.95,328.86,331.54,313.35,320.28,
        293.27,180.05,269.84,340.41,266.42,352.33,346.80,384.27,417.02,437.73,
        431.12,378.27,315.29,376.35
    ],
    "Coffee_price": [
        0.91,0.86,0.65,0.69,1.45,1.46,1.16,1.79,1.29,1.04,
        0.90,0.56,0.54,0.62,0.77,1.08,1.08,1.18,1.33,1.26,
        1.64,2.53,1.75,1.26,1.78,1.33,1.37,1.33,1.14,1.02,
        1.11,1.69,2.14,1.70
    ],
    "Sugar_price": [
        0.13,0.09,0.09,0.10,0.12,0.12,0.11,0.11,0.09,0.06,
        0.08,0.08,0.06,0.07,0.07,0.10,0.15,0.10,0.12,0.18,
        0.22,0.27,0.22,0.17,0.16,0.13,0.18,0.16,0.12,0.12,
        0.13,0.18,0.19,0.24
    ],
    "Tea_price": [
        0.92,0.84,0.91,0.84,0.83,0.75,0.81,1.08,1.08,1.06,
        1.13,0.90,0.81,0.88,0.90,0.98,1.10,0.96,1.23,1.43,
        1.44,1.57,1.59,1.21,1.08,1.55,1.31,1.65,1.36,1.22,
        1.15,1.21,1.26,1.32
    ],
    "Annual_income": [
        42650,43240,44220,47220,49340,51350,53680,56900,59590,62570,
        65770,66860,66970,68560,70390,73300,77320,78850,79630,78540,
        78180,81010,82840,87670,88770,92670,97360,103200,106000,116700,
        115300,121800,126500,135700
    ]
}

# Create a Pandas dataframe
df = pd.DataFrame(data)

# Display first few rows
print(df.head())

# The data are stored in a Pandas DataFrame, which resembles an Excel spreadsheet. Variable names appear as column headers, and each row represents an observation. The head() command displays the first five observations. It allows us to quickly verify that the data were entered correctly. If we wanted the first 20 observations, we put 20 into the parenthesis. Lastly, if we wanted the end of the dataset, we use .tail(). It functions similarly to .head().
# 
# Next, we compute descriptive statistics to summarize the dataset. These statistics provide insight into the distribution, variability, and shape of each variable.
# 
# We first create a copy of the dataset and remove the Year variable. Although time is important for later analysis, it is not a variable in the economic sense. It should be excluded in summary statistics.
# 
# We then compute the minimum, average, maximum, standard deviation, skewness, and kurtosis. Storing these results in a DataFrame produces a clean, tabular output.
# 

# In[2]:

# Descriptive statistics

# Create a copy of the dataframe
stats = df.copy()

# Remove the year
stats.drop("Year", axis=1, inplace=True)

# Compute statistics
desc_stats = pd.DataFrame({
    "Minimum": stats.min(),
    "Average": stats.mean(),
    "Maximum": stats.max(),
    "Std Dev": stats.std(),
    "Skewness": stats.skew(),
    "Kurtosis": stats.kurt()
})

# Round for nicer display (optional)
desc_stats = desc_stats.round(2)

# Print results
print("Descriptive Statistics")
print(desc_stats.T)

# After summarizing each variable individually, we examine how variables move together using a correlation matrix. As before, we remove the Year variable to avoid treating time as an explanatory variable.

# In[3]:

# Correlation matrix

# Remove Year variable
stats = df.drop("Year", axis=1)

# Compute correlation matrix
corr_matrix = stats.corr()

# Round for nicer display
corr_matrix = corr_matrix.round(2)

# Display results
print("Correlation Matrix")
print(corr_matrix)


# In this section, we construct two commonly used plots: box plots and scatter plots. We begin with box-and-whisker plots, which provide a compact visual summary of a variable’s distribution. A box plot displays the median, dispersion, and potential outliers in a single figure.
# 
# The code structure follows the same logic as before. We create a new DataFrame that includes only the variables of interest. In this case, we exclude both Year and Annual_income. The Year variable is not meaningful for distributional analysis. In addition, Annual_income is measured on a much larger scale and would dominate the visualization. Removing it allows us to present the remaining variables clearly.
# 

# In[4]:

# Import library
import matplotlib.pyplot as plt

# Remove Year variable
stats = df.drop( ["Year","Annual_income"], axis=1)

# Create subplots (one for each variable)
fig, axes = plt.subplots(nrows=2, ncols=2)

# Flatten axes for easy looping
axes = axes.flatten()

# Loop through variables and plot
for i, col in enumerate(stats.columns):
    axes[i].boxplot(stats[col])
    axes[i].set_title(col)

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()

# Next, we examine the relationship between coffee price and coffee consumption using a scatter plot. This visualization allows us to assess whether the data are consistent with the Law of Demand, which predicts an inverse relationship between price and quantity demanded.

# In[5]:

# Import libraries
import matplotlib.pyplot as plt
import numpy as np

# Define variables (switched axes)
x = df["Coffee_consumption"]
y = df["Coffee_price"]

# Create scatter plot
plt.figure()
plt.scatter(x, y, color="cornflowerblue")

# Add regression line
m, b = np.polyfit(x, y, 1)

# Sort for smooth line
sorted_idx = np.argsort(x)
x_sorted = x.iloc[sorted_idx]
y_pred = m * x_sorted + b

plt.plot(x_sorted, y_pred, color="crimson")

# Labels and title
plt.xlabel("Coffee Consumption")
plt.ylabel("Coffee Price")
plt.title("Coffee Price vs Coffee Consumption")

# Grid
plt.grid()

# Show plot
plt.show()

# In[ ]:
