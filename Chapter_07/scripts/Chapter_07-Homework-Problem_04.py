# Chapter 7-Homework-Problem 4

# A retail company wants to better understand its customers in order to improve marketing strategies. The company collected data on customer spending behavior but does not know how customers should be grouped.
# 
# Our task is to use cluster analysis to identify natural groupings of customers.
# 
# Each observation represents one customer. The variables include:
# •	Age – customer’s age
# •	Annual Income (k$) – annual income in thousands of dollars
# •	Spending Score (1–100) – a score assigned by the store based on spending behavior
# 
# a. We explore the data, display the first five observations, and compute summary statistics.
# 
# b. We create a scatter plot of annual income vs spending score. What patterns do we observe?
# 
# c. We determine the number of clusters. We use the Elbow Method to identify the optimal number of clusters.
# 
# d. We plot the clusters using a scatter plot. We use different colors for each cluster.
# 
# e. We describe each cluster in plain language. For example: “High income, high spending.” How could the company use these clusters for marketing?
# 

# In[1]:

# ============================================
# Cluster Analysis: Customer Segmentation
# ============================================

# --------------------------------------------
# Step 1: Import required libraries
# --------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans

# --------------------------------------------
# Step 2: Load the dataset
# --------------------------------------------

df = pd.read_csv('Chapter_07-mall_customers.csv')

# --------------------------------------------
# Step 3: Explore the data
# --------------------------------------------
print("First five observations:")
print(df.head())

print("\nSummary statistics:")
print(df.describe())

# --------------------------------------------
# Step 4: Visualize the data
# --------------------------------------------
plt.figure()
sns.scatterplot(
    data=df,
    x='Annual Income (k$)',
    y='Spending Score (1-100)'
)

plt.title('Customer Distribution')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.show()

# --------------------------------------------
# Step 5: Determine number of clusters (Elbow Method)
# --------------------------------------------
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

inertia = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plot Elbow curve
plt.figure()
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()

# --------------------------------------------
# Step 6: Apply K-means using chosen k
# --------------------------------------------
# Based on the elbow plot, students select k (typically k = 5)
optimal_k = 5

kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# --------------------------------------------
# Step 7: Visualize clusters
# --------------------------------------------
plt.figure()
sns.scatterplot(
    data=df,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',
    palette='Set2'
)

plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.legend(title='Cluster')
plt.show()

# --------------------------------------------
# Step 8: Examine cluster characteristics
# --------------------------------------------
print("\nCluster Summary:")
print(df.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean())

# In[ ]:
