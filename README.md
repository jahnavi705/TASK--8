# TASK--8
 Clustering with K-Means

# Objective :
 Perform unsupervised learning using the K-Means clustering algorithm to group customers based on their purchasing behavior.

# Tools & Libraries :
 - Python
 - Scikit learn
 - Pandas
 - Matplotlib
 - StandardScaler
 - Silhouette Score

# Steps :
 1. Load the Preprocess the Data
  - Loaded the dataset using Pandas
  - Selected key features : Annual Income and Spending Score (1-100)
  - Applied StandardScaler to normalize the features
    
 2. Determine Optimal Clusters with Elbow Method
  - Used K-Means clustering with cluster values from 1 to 10
  - Computed WCSS (Within Cluster Sum of Squares)
  - Plotted the Elbow graph to visually datermine the optimal number of clusters(K)
    
 3. Apply K-Means Clustering
  - Fitted K Means with the chosen number of cluster
  - Assigned cluster labels to each data point
 4. Visualize the Clusters
  - Created a scatter plot with color-coded clusters
  - Marked cluster centroids using black 'X' markers
 5. Evaluate Clustering
  - Calculated the Silhouette Score to measure cluster separation and cohesion

# output :

![pic](https://github.com/user-attachments/assets/61f2c3ed-e51a-465a-9e40-958550e19d66)

![picc](https://github.com/user-attachments/assets/d7e1d12e-61f8-46d6-ba2c-34c4b6039d9c)
