# Chapter 4-Homework-Exercises

# Problem 1.
# 
# We visit the End-of-Chapter Problems from Chapter 1. An investor is deciding whether to diversify their portfolio by investing in two sectors: technology and energy. The annual returns (%) for two representative stocks over five years are given in the Python code below.
# 
# a. Please compute the following for each stock: minimum, mean, maximum, standard deviation, skewness, and kurtosis.
# 
# b. Compute the correlation coefficient between the two stocks.
# 
# c. Based on your results:<br />
# •	Which stock offers higher average returns?<br />
# •	Which stock is riskier?<br />
# •	Does diversification appear beneficial?

# In[1]:

# Libraries
import pandas as pd

# Create the dataset using pandas
data = {
    "Year": [1, 2, 3, 4, 5],
    "Tech": [10, 18, -6, 20, 9],
    "Energy": [4, 12, -3, 8, 7]
}

df = pd.DataFrame(data)

# Display the dataset
print("Dataset:")
print(df)
print()

# Descriptive statistics
print("Descriptive Statistics:")
print(df[["Tech", "Energy"]].agg(["mean", "min", "max", "std","skew","kurt"]))
print()

# Correlation
correlation = df["Tech"].corr(df["Energy"])

print("Correlation between Tech and Energy:")
print(correlation)

# Problem 2
#     
# A national survey reports that adults spend an average of 6.5 hours per day on screens. A researcher suspects that remote workers spend more time on screens than this average. A sample of 36 remote workers shows an average screen time of 7.1 hours per day. Assume the population standard deviation is 1.8 hours.
# 
# a. Compute the test statistic.
# 
# b. Compute the p-value.
# 
# c. At α = 0.05, state your conclusion.
# 

# In[2]:

# Import the library
import math
from scipy.stats import norm

# Given data
x_bar = 7.1      # sample mean
mu = 6.5         # population mean
sigma = 1.8      # population standard deviation
n = 36           # sample size
alpha = 0.05     # significance level

# Step 1: Compute the test statistic (z)
z = (x_bar - mu) / (sigma / math.sqrt(n))

print("Z-statistic:", round(z, 2))

# Step 2: Compute the p-value (right-tailed test)
p_value = 1 - norm.cdf(z)

print("P-value:", round(p_value, 3))

# Step 3: Conclusion
if p_value < alpha:
    print("Reject the null hypothesis.")
else:
    print("Fail to reject the null hypothesis.")

# Problem 3.
# 
# A technology company wants to compare the average battery life of two smartphone models.
# 
# Model A uses a new battery design<br />
# Model B uses the current standard battery
# 
# The company wants to know whether the mean battery life for the new design lasts longer than the standard battery. Table 4.3 contains the data.
# <table>
#     <tr colspan="2"><td>
#         Table 4.3. Battery Life Data</td>
#     </tr>
#     <tr>
#         <td>Model A (New Battery)</td><td>Model B (Standard Battery)</td>
#     </tr>
#     <tr>
#         <td>n1 = 36</td><td>n2 = 49</td>
#     </tr>
#     <tr>
#         <td>xbar 1 = 11.8 hours</td><td>xbar 2 = 10.9 hours</td>
#     </tr>
#     <tr>
#         <td>σ1 = 2.4 hours</td><td>σ2 = 2.8 hours</td>
#     </tr>
# </table>
# 
# a. State the hypotheses
# 
# b. Compute the test statistic
# 
# c. Compute the p-value
# 
# d. What is our conclusion using α = 0.05?
# 

# In[3]:

# Import libraries
import math
from scipy.stats import norm

# Given data
x1_bar = 11.8   # sample mean (Model A - new battery)
x2_bar = 10.9   # sample mean (Model B - standard battery)

sigma1 = 2.4    # population standard deviation (Model A)
sigma2 = 2.8    # population standard deviation (Model B)

n1 = 36         # sample size (Model A)
n2 = 49         # sample size (Model B)

alpha = 0.05

# Step 1: Compute the standard error
std_error = math.sqrt((sigma1**2)/n1 + (sigma2**2)/n2)

print("Standard Error:", round(std_error, 3))

# Step 2: Compute the test statistic (z)
z = (x1_bar - x2_bar) / std_error

print("Z-statistic:", round(z, 2))

# Step 3: Compute the p-value (RIGHT-tailed test)
p_value = 1 - norm.cdf(z)

print("P-value:", round(p_value, 4))

# Step 4: Conclusion
if p_value < alpha:
    print("Reject the null hypothesis.")
else:
    print("Fail to reject the null hypothesis.")

# Problem 4.
# 
# A university wants to compare the average exam scores of two teaching methods.
# 
# Method A uses traditional lectures<br />
# Method B uses interactive learning
# 
# The university wants to determine whether the mean scores differ between the two methods. Table 4.4 shows the summary statistics.
# <table>
#     <tr colspan="2"><td>
#         Table 4.4. Teaching Method Data</td>
#     </tr>
#     <tr>
#         <td>Method A (Traditional)</td><td>Method B (Interactive)</td>
#     </tr>
#     <tr>
#         <td>n1 = 25</td><td>n2 = 30</td>
#     </tr>
#     <tr>
#         <td>xbar 1 = 78.6</td><td>xbar 2 = 74.2</td>
#     </tr>
#     <tr>
#         <td>s1 = 10.5</td><td>s2 = 9.8</td>
#     </tr>
# </table>
# 
# a. State the hypotheses
# 
# b. Compute the test statistic
# 
# c. Compute the p-value
# 
# d. What is our conclusion using α = 0.05?

# In[4]:

# --------------------------------------------------
# Two-Sample t-Test (Welch's Test)
# Comparing Teaching Methods
# --------------------------------------------------

# Step 1: Import libraries
import numpy as np
from scipy import stats

# --------------------------------------------------
# Step 2: Input sample statistics
# --------------------------------------------------

# Method A (Traditional)
n1 = 25
x1_bar = 78.6
s1 = 10.5

# Method B (Interactive)
n2 = 30
x2_bar = 74.2
s2 = 9.8

# --------------------------------------------------
# Step 3: State the hypotheses
# --------------------------------------------------

print("H0: μ1 = μ2")
print("Ha: μ1 ≠ μ2")

# --------------------------------------------------
# Step 4: Compute test statistic
# --------------------------------------------------

# Standard error
se = np.sqrt((s1**2 / n1) + (s2**2 / n2))

# t-statistic
t_stat = (x1_bar - x2_bar) / se

print("\nt-statistic:", t_stat)

# --------------------------------------------------
# Step 5: Degrees of freedom (Welch approximation)
# --------------------------------------------------

df = ((s1**2 / n1 + s2**2 / n2)**2) / \
     (((s1**2 / n1)**2 / (n1 - 1)) + ((s2**2 / n2)**2 / (n2 - 1)))

print("Degrees of freedom:", df)

# --------------------------------------------------
# Step 6: Compute p-value (two-tailed test)
# --------------------------------------------------

p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

print("p-value:", p_value)

# --------------------------------------------------
# Step 7: Decision
# --------------------------------------------------

alpha = 0.05

if p_value < alpha:
    print("Reject H0: The mean scores differ between the two methods.")
else:
    print("Fail to reject H0: No significant difference in mean scores.")

# Problem 5.
# 
# A retail company is working to improve the consistency of its customer experience. Management believes that reducing variability in customer satisfaction is just as important as increasing the average score.
# 
# Customer satisfaction is measured on a 100-point scale. Based on past data, the company expects a population standard deviation of σ = 10.
# 
# After implementing a new employee training program, the company collects a sample of 15 customer satisfaction scores. The Python code holds the 15 observations.
# 
# a. What is the sample mean customer satisfaction score?
# 
# b. What is the sample variance?
# 
# c. What is the sample standard deviation?
# 
# d. Conduct a hypothesis test to determine whether the variability has changed after the training program. Use a significance level of 0.05.<br />
# •	State the null and alternative hypotheses<br />
# •	Compute the test statistic<br />
# •	State the decision rule<br />
# •	Provide your conclusion
# 

# In[5]:

# Libraries
import pandas as pd
from scipy.stats import chi2

# Create the dataset
data = {
    "Score": [88, 84, 91, 79, 85,
              87, 82, 90, 86, 83,
              89, 81, 92, 78, 84]
}

df = pd.DataFrame(data)

# Descriptive statistics
print("Descriptive Statistics:")
print(df["Score"].agg(["mean", "var", "std"]))
print()

# Extract values for hypothesis test
alpha = 0.05
n = len(df)
s2 = df["Score"].var()
sigma2 = 100

# Right-tail critical value
critical_value = chi2.ppf(1 - alpha, n-1)

print("Critical chi-square value:")
print(critical_value)
print(" ")

# Chi-square test statistic
chi_square = (n - 1) * s2 / sigma2

print("Chi-square test statistic:")
print(chi_square)

# Problem 6.
# 
# A manufacturing company operates two production lines that produce the same product. Management wants to ensure that both lines maintain consistent quality. In particular, they are concerned about variability in product thickness, since high variability can lead to defects.
# 
# To evaluate consistency, the company collects samples from each production line:
# 
# Production Line 1: n1 = 21, s_1^2 = 8.2<br />
# Production Line 2: n2 = 26, s_2^2 = 3.0
# 
# Management wants to test whether the variability of the two production lines is the same. Please use Python to solve this problem. The hypotheses are below:
# 
# H0: σ1<sup>2</sup> = σ2<sup>2</sup><br />
# Ha: σ1<sup>2</sup> ≠ σ2<sup>2</sup>
# 
# a. Use the p-value approach to test the hypotheses at the 5% significance level.
# 
# Hint: Place the larger sample variance in the numerator and conduct a right-tailed test.
# 
# b. Repeat the test using the critical value approach.
# 

# In[6]:

# Load the library
from scipy.stats import f

# Data
n1 = 21
s1_sq = 8.2

n2 = 26
s2_sq = 3.0

alpha = 0.05

# Step 1: Put larger variance in numerator
F = s1_sq / s2_sq

print("F-statistic:")
print(F)
print()

# Step 2: Degrees of freedom
df1 = n1 - 1
df2 = n2 - 1

print("Degrees of freedom:")
print("df1 = ", df1)
print("df2 = ", df2)
print()

# Right-tail probability
p_value_right = 1 - f.cdf(F, df1, df2)

# Two-tailed p-value
p_value = 2 * p_value_right

print("p-value:")
print(p_value)

# Two-tailed test use alpha/2 in right tail
critical_value = f.ppf(1 - alpha/2, df1, df2)

print("Critical F-value:")
print(critical_value)
print()

# In[ ]: