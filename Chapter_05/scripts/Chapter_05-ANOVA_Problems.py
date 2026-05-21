# ANOVA Problems in Chapter 5

# In Example 1, Magic Wax tests three types of car wax—Type 1, Type 2, and Type 3—to compare durability. Each wax is applied to five cars, and durability is measured by the number of washes the wax withstands before deterioration.
# 
# We test whether all waxes have the same average durability.

# Import libraries
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Step 1: Create the dataset
data = {
    "Type 1": [28, 29, 30, 28, 32],
    "Type 2": [32, 28, 32, 29, 31],
    "Type 3": [30, 27, 31, 31, 32]
}

# Create a Pandas dataframe
df = pd.DataFrame(data)

# Step 2: Convert to long format
df_long = df.melt(var_name="Wax", value_name="Durability")

# Step 3: Fit the ANOVA model
model = ols('Durability ~ C(Wax)', data=df_long).fit()

# Step 4: Get ANOVA table
anova_table = sm.stats.anova_lm(model, typ=2)

# Step 5: Print results
print("\n=== ANOVA Table ===")
print(anova_table)

# In Example 2, Lazer Manufacturing investigates whether managers at three plants—Chicago, St. Louis, and Detroit—work different numbers of hours per week. A random sample of five managers is selected from each plant.

# ============================================================
# ANOVA and Fisher's LSD Post-Hoc Test Example
# ============================================================

# Import libraries
# pandas --> used for creating and managing data tables
# statsmodels --> used for statistical models and ANOVA
# ols --> ordinary least squares regression model
# combinations --> creates all possible pairwise comparisons

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from itertools import combinations
import scipy.stats as stats
import math

# ============================================================
# Step 1: Create the Dataset
# ============================================================

# The data represent production output from three plants.
# Each list contains five observations for each plant.

data = {
    "Chicago": [49, 53, 58, 62, 53],
    "StLouis": [74, 62, 67, 74, 63],
    "Detroit": [50, 64, 60, 56, 55]
}

# Convert the dictionary into a Pandas DataFrame.
# A DataFrame is a table structure similar to Excel.

df = pd.DataFrame(data)

# Print the original dataset
print("\n=== Original Dataset ===")
print(df)

# ============================================================
# Step 2: Convert Data from Wide Format to Long Format
# ============================================================

# ANOVA in statsmodels requires the data in "long format."
#
# Wide format:
# Chicago   StLouis   Detroit
#
# Long format:
# Plant      Output
# Chicago      49
# Chicago      53
# StLouis      74
# etc.

# var_name --> name of the categorical variable
# value_name --> name of the numerical variable

df_long = df.melt(var_name="Plant", value_name="Output")

# Print the transformed dataset
print("\n=== Long Format Dataset ===")
print(df_long)

# ============================================================
# Step 3: Fit the ANOVA Model
# ============================================================

# OLS stands for Ordinary Least Squares.
#
# The model formula:
# Output ~ C(Plant)
#
# Output --> dependent variable
# Plant --> independent categorical variable
#
# C(Plant) tells Python that Plant is a categorical variable.

model = ols('Output ~ C(Plant)', data=df_long).fit()

# ============================================================
# Step 4: Generate the ANOVA Table
# ============================================================

anova_table = sm.stats.anova_lm(model, typ=2)

# Print the ANOVA table
print("\n=== ANOVA Table ===")
print(anova_table)


# In this case, the p-value is below α = 0.05. We reject the null hypothesis and conclude that at least one plant has a different average number of hours worked. To identify which groups differ, we conduct pairwise comparisons.

# ============================================================
# Step 5: Fisher's LSD Post-Hoc Test
# ============================================================

# unique() extracts all plant names
groups = df_long['Plant'].unique()

# combinations(groups, 2)
# creates all possible pairs:
# Chicago vs StLouis
# Chicago vs Detroit
# StLouis vs Detroit

pairs = list(combinations(groups, 2))

# Print section heading
print("\n=== Fisher's LSD Pairwise Comparisons ===")

# Degrees of freedom for error
df_error = anova_table.loc['Residual', 'df']

# Get the sum of squares from the ANOVA and calculate MSE
mse = anova_table.loc['Residual', 'sum_sq'] / df_error

# Significance level
alpha = 0.05

# Get the two-tailed critical t-value
t_critical = stats.t.ppf(1 - alpha/2, df_error)

# Loop through each pair of plants
for g1, g2 in pairs:

    # --------------------------------------------------------
    # Extract observations for each group
    # --------------------------------------------------------

    # Select Output values where Plant == g1
    data1 = df_long.loc[df_long['Plant'] == g1, 'Output']

    # Select Output values where Plant == g2
    data2 = df_long.loc[df_long['Plant'] == g2, 'Output']

    # Count the number of observations
    n1 = data1.count()

    # Count the number of observations
    n2 = data2.count()

    # --------------------------------------------------------
    # Calculate group means, LSD, and make comparison
    # --------------------------------------------------------

    mean1 = data1.mean()
    mean2 = data2.mean()

    # Calculate difference in means
    mean_diff = abs( mean1 - mean2 )

    # Calculate the LSD
    lsd = t_critical * math.sqrt(mse * (1/n1 + 1/n2))

    # Compare LSD with the mean difference
    if mean_diff > lsd:
        decision = "reject"
    else:
        decision = "do not reject"

    # --------------------------------------------------------
    # Print comparison results
    # --------------------------------------------------------

    print(f"{g1} vs {g2}: mean_diff = {mean_diff:.3f}, LSD = {lsd:.3f}, Decision: {decision}")


# Step 6: Tukey (HSD) pairwise comparisons
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Perform pairwise comparisons (LSD-style interpretation)
hsd = pairwise_tukeyhsd(endog=df_long["Output"],
                        groups=df_long["Plant"],
                        alpha=0.05)

print("\n=== Pairwise Comparisons (LSD) ===")
print(hsd)


# In Example 3, New Oil Company tests three gasoline blends (X, Y, and Z) using the same vehicles. Because the same cars are used across treatments, each car acts as a block, controlling for vehicle-specific effects.

# Import libraries
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Step 1: Create the dataset
data = {
    "Block": [1,2,3,4,5]*3,
    "Treatment": ["X"]*5 + ["Y"]*5 + ["Z"]*5,
    "Output": [
        13.2,12.8,12.3,14.0,11.1,   # Blend X
        12.8,12.3,12.3,13.2,10.6,   # Blend Y
        12.8,12.3,11.9,12.3,11.1    # Blend Z
    ]
}

# Create a Pandas dataframe
df = pd.DataFrame(data)

# Step 2: (Already in long format)
df_long = df

# Step 3: Fit the ANOVA model (with blocks)
model = ols('Output ~ C(Treatment) + C(Block)', data=df_long).fit()

# Step 4: Get ANOVA table
anova_table = sm.stats.anova_lm(model, typ=1)

# Step 5: Print results
print("\n=== ANOVA Table (With Blocks) ===")
print(anova_table)


# In Example 4, PowerClean Company studies how cleaning performance depends on:<br />
# •	Detergent type (Standard vs. Premium)<br />
# •	Water temperature (Cold, Warm, Hot)
# 
# Each combination is tested twice. It allows us to estimate interaction effects.
# 
# Import libraries
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Step 1: Create the dataset
data = {
    "Detergent": ["Standard"]*6 + ["Premium"]*6,
    "Temperature": ["Cold","Warm","Hot"]*4,
    "Output": [
        65,72,78, 67,74,80,   # Standard (2 reps)
        66,75,88, 69,77,90    # Premium (2 reps)
    ]
}

# Create a Pandas dataframe
df = pd.DataFrame(data)

# Step 2: (Already in long format)
df_long = df

# Step 3: Fit the ANOVA model (with interaction)
model = ols('Output ~ C(Detergent) + C(Temperature) + C(Detergent):C(Temperature)', data=df_long).fit()

# Step 4: Get ANOVA table
anova_table = sm.stats.anova_lm(model, typ=1)

# Step 5: Print results
print("\n=== ANOVA Table (Two-Way with Interaction) ===")
print(anova_table)
