# ANOVA Problems in Chapter 5

# In Example 1, Magic Wax tests three types of car wax—Type 1, Type 2, and Type 3—to compare durability. Each wax is applied to five cars, and durability is measured by the number of washes the wax withstands before deterioration.
# 
# We test whether all waxes have the same average durability.
# 

# In[1]:

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

# In[2]:

# Import libraries
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Step 1: Create the dataset
data = {
    "Chicago": [49,53,58,62,53],
    "StLouis": [74,62,67,74,63],
    "Detroit": [50,64,60,56,55]
}

# Create a Pandas dataframe
df = pd.DataFrame(data)

# Step 2: Convert to long format
df_long = df.melt(var_name="Plant", value_name="Output")

# Step 3: Fit the ANOVA model
model = ols('Output ~ C(Plant)', data=df_long).fit()

# Step 4: Get ANOVA table
anova_table = sm.stats.anova_lm(model, typ=2)

# Step 5: Print results
print("\n=== ANOVA Table ===")
print(anova_table)

# In this case, the p-value is below α = 0.05. We reject the null hypothesis and conclude that at least one plant has a different average number of hours worked. To identify which groups differ, we conduct pairwise comparisons.

# In[3]:


# Step 6: Least Significant Difference (LSD) pairwise comparisons
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Perform pairwise comparisons (LSD-style interpretation)
lsd = pairwise_tukeyhsd(endog=df_long["Output"],
                        groups=df_long["Plant"],
                        alpha=0.05)

print("\n=== Pairwise Comparisons (LSD) ===")
print(lsd)


# In Example 3, New Oil Company tests three gasoline blends (X, Y, and Z) using the same vehicles. Because the same cars are used across treatments, each car acts as a block, controlling for vehicle-specific effects.

# In[4]:

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

# In[5]:

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


# In[ ]:
