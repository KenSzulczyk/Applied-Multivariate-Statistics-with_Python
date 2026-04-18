# Chapter 4 Python Exercises

# <b>4.2 Setup and Environment</b>
# 
# The first line prints the classic message “Hello World,” which has been used since the 1980s to introduce programming. Next, we assign the value 1 to a variable named x. The if statement then checks whether x is equal to 1. If the condition is true, Python prints the message “x is 1.”

# In[1]:

print("Hello World")

x = 1
if x == 1:
    print("x is 1")

# Let’s look at a simple example.
# 
# •	The first line assigns the value 5 to the variable x. This is an integer.<br />
# •	The second line assigns 5.0 to y, which is a floating-point number.<br />
# •	The third line assigns the text “John” to the variable name, which is a string.
# 
# To see the type of each variable, we use the type() function.
# 

# In[2]:

x = 5
y = 5.0
name = "John"

print(type(x))
print(type(y))
print(type(name))

# Python lists are indexed starting at zero, which means:
# 
# •	The first element has index 0<br />
# •	The second element has index 1<br />
# •	And so on
# 
# We can use a loop to go through each value in a list. A loop allows us to repeat an action for every element. In this case, we will use a for loop to read each number in the list and print it.

# In[3]:

mylist = [1,2,3,4,5]

for x in mylist:
    print(x)

# Python supports standard arithmetic operations such as addition, subtraction, multiplication, division, and exponents. These operators allow us to perform calculations quickly and efficiently.
# 
# In the first example, we calculate 2<sup>3</sup> using the ** operator, which represents an exponent. In the second example, we divide 10 by 3 and use the % operator to find the remainder, which is also called the modulus.
# 

# In[4]:

x = 2 ** 3
y = 10 % 3
print(x, y)

# First, we create a list of data values. Then, we calculate:
# 
# •	the mean (average) using statistics.mean()<br />
# •	the standard deviation using statistics.stdev()
# 
# The notation statistics.mean tells Python to use the mean function from the statistics library.
# 
# Finally, we print the results using an f-string. The {} brackets allow us to insert variable values directly into the text.
# 

# In[5]:

import statistics

data = [10, 13, 19, 18, 17, 24, 20, 15]

# Calculate the mean (sample arithmetic mean)
mean_value = statistics.mean(data)
# Calculate the standard deviation (sample standard deviation)
std_deviation = statistics.stdev(data)

print(f"Mean of the data: {mean_value}")
print(f"Standard Deviation of the data: {std_deviation}")

# Let’s look at an example.<br />
# •	The first line assigns the text "Hello World" to the variable text.<br />
# •	The len() function counts how many characters are in the string, including spaces.<br />
# •	The upper() method converts all letters to uppercase.<br />
# •	The final command selects the first five characters of the string.
# 
# Remember, Python uses zero-based indexing, so counting starts at 0. This means:<br />
# •	index 0 is the first character<br />
# •	index 4 is the fifth character
# 

# In[6]:

text = "Hello World"
print(len(text))
print(text.upper())
print(text[0:5])

# A condition evaluates an expression and returns either True or False. Based on this result, Python controls the flow of the program by deciding which block of code to execute.
# 
# In the example below, we create a condition to check whether x equals 2. It is important to distinguish between the assignment operator and the equality operator. A single equal sign (=) assigns a value to a variable, while two equal signs (==) test whether two values are equal.
# 
# If the condition evaluates to True, Python executes the code inside the if block. If the condition evaluates to False, Python skips that block and executes the code inside the else block instead.
# 

# In[7]:

x = 2
if x == 2:
    print("x equals 2")
else:
    print("x does not equal 2")

# Python provides two main types of loops: the for loop and the while loop.
# 
# A for loop is commonly used to iterate over a sequence of values, such as numbers, lists, or strings. In many cases, we use the range() function to specify how many times the loop should run.
# 
# A while loop works differently. It continues executing as long as a specified condition remains True. Once the condition becomes False, the loop stops.
# 
# In the example below, the for loop prints the numbers from 0 to 4. The range(5) function generates five values: 0, 1, 2, 3, and 4.
# 
# The while loop also prints numbers from 100 to 104. It starts with count = 100 and continues looping as long as count < 104. Inside the loop, the variable count increases by 1 each time using count += 1.
# 

# In[8]:

for i in range(5):
    print(i)

count = 100
while count < 104:
    print(count)
    count += 1

# A function takes inputs, performs a task, and can return an output. The inputs are called parameters, and the output is specified using the return statement.
# 
# In the example below, we define a function called add that takes two parameters, a and b. The function adds these two values together and returns the result.
# 
# We then call the function by passing in the numbers 2 and 3. The function computes the sum and returns 5.

# In[9]:

def add(a,b):
    return a + b

print(add(2,3))

# In the example below, we use a dictionary to store names and phone numbers in an object called phonebook. Each name (the key) is linked to a phone number (the value).
# 
# To access a value, we place the key inside square brackets. In this case, we insert "John" into the dictionary, and Python returns John’s phone number.
# 

# In[10]:

phonebook = {"John":123456, "Jane":456554, "Ken": 484545}

print("John phone number is: ", phonebook["John"])

# To use a library, we must first import it. In the example below, we import NumPy and Matplotlib. We assign them shorter names, called aliases, using the as keyword. This allows us to write less code and improves readability.
# 
# •	numpy becomes np<br />
# •	matplotlib.pyplot becomes plt
# 
# The linspace() function from NumPy creates 100 evenly spaced values between 0 and 50. We then use Matplotlib’s plot() function to graph x and x2. Finally, the show() function tells Python to display the graph. In some environments, this command is necessary; otherwise, the graph may not appear.
# 
# We also use the # symbol to insert comments. Comments are ignored by Python but are extremely important for explaining what the code does. When we return to our code later, comments help us and others quickly understand our logic.
# 

# In[11]:

import numpy as np
import matplotlib.pyplot as plt

# Create 200 x values between 0 and 100
x = np.linspace(0,100,200)

# Plot the graph of x and x squared
plt.plot(x, x**2)

plt.show()

# We begin with the dataset on temperature and electricity demand from Chapter 1, shown in Table 4.1. A city energy planner is interested in understanding how weather conditions affect electricity usage. To explore this relationship, the planner collects data on average daily temperature and electricity demand over five days.
# 
# We compute the descriptive statistics and correlation using Python.

# In[12]:

# --------------------------------------------------
# Temperature and Electricity Demand Analysis
# --------------------------------------------------

# Step 1: Import libraries
import pandas as pd

# --------------------------------------------------
# Step 2: Manually create the dataset
# --------------------------------------------------

data = {
    "Day": [1, 2, 3, 4, 5],
    "Temperature": [22, 25, 28, 31, 27],
    "Electricity_Demand": [210, 250, 290, 330, 270]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Display dataset
print("Dataset:")
print(df)

# --------------------------------------------------
# Step 3: Descriptive Statistics
# --------------------------------------------------

print("\nDescriptive Statistics:")

# Minimum values
print("\nMinimum:")
print(df[["Temperature", "Electricity_Demand"]].min())

# Mean values
print("\nMean:")
print(df[["Temperature", "Electricity_Demand"]].mean())

# Maximum values
print("\nMaximum:")
print(df[["Temperature", "Electricity_Demand"]].max())

# Standard deviation
print("\nStandard Deviation:")
print(df[["Temperature", "Electricity_Demand"]].std())

# --------------------------------------------------
# Step 4: Pearson Correlation
# --------------------------------------------------

correlation = df["Temperature"].corr(df["Electricity_Demand"])

print("\nPearson Correlation:")
print(correlation)

# Next, we revisit the hypothesis testing problem involving the lifetimes of two types of light bulbs. The summary statistics are shown in Table 4.2.
# 
# <table>
#     <tr colspan="2"><td>
#         Table 4.2. Bulb Lifetime Comparison</td>
#     </tr>
#     <tr>
#         <td>Brand A (Energy Efficient)</td><td>Brand B (Standard)</td>
#     </tr>
#     <tr>
#         <td>n1 = 30</td><td>n2 = 35</td>
#     </tr>
#     <tr>
#         <td>xbar 1 = 1,240</td><td>xbar 2 = 1,180</td>
#     </tr>
#     <tr>
#         <td>s1 = 120</td><td>s2 = 150</td>
#     </tr>
# </table>
#     
# We first state the hypotheses. We test whether Brand A has a longer lifetime than Brand B:
# 
# H0: mu1 <= mu2<br />
# Ha: mu1 > mu2
# 
# We use Python to perform Welch’s two-sample t-test based on summary statistics.

# In[13]:

# --------------------------------------------------
# Two-Sample t-Test (Welch's Test)
# Comparing Bulb Lifetimes
# --------------------------------------------------

# Step 1: Import libraries
import numpy as np
from scipy import stats

# --------------------------------------------------
# Step 2: Input sample statistics
# --------------------------------------------------

n1 = 30
x1_bar = 1240
s1 = 120

n2 = 35
x2_bar = 1180
s2 = 150

# --------------------------------------------------
# Step 3: Compute test statistic
# --------------------------------------------------

# Standard error
se = np.sqrt((s1**2 / n1) + (s2**2 / n2))

# t-statistic
t_stat = (x1_bar - x2_bar) / se

print("t-statistic:", t_stat)

# --------------------------------------------------
# Step 4: Degrees of freedom (Welch formula)
# --------------------------------------------------

df = ((s1**2 / n1 + s2**2 / n2)**2) / \
     (((s1**2 / n1)**2 / (n1 - 1)) + ((s2**2 / n2)**2 / (n2 - 1)))

print("Degrees of freedom:", df)

# --------------------------------------------------
# Step 5: Compute p-value (one-tailed test)
# --------------------------------------------------

p_value = 1 - stats.t.cdf(t_stat, df)

print("p-value:", p_value)

# --------------------------------------------------
# Step 6: Decision
# --------------------------------------------------

alpha = 0.05

if p_value < alpha:
    print("Reject H0: Brand A lasts longer than Brand B.")
else:
    print("Fail to reject H0: Not enough evidence that Brand A lasts longer.")

# In[ ]:




