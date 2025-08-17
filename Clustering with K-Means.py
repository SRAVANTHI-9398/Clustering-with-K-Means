# Task 8: Clustering with K-Means

# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# 1. Load dataset (Mall Customer Segmentation dataset as example)
# Replace 'Mall_Customers.csv' with your dataset path
df = pd.read_csv("Mall_Customers.csv")

print("First 5 rows of data:")
print(df.head())

# Select relevant features (e.g., Annual Income and Spending Score)
X = df.iloc[:, [3, 4]].values   # here using columns "Annual Income" & "Spending Score"

# 2. Elbow Method to find optimal K
wcss = []   # within-cluster sum of squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.show()

# 3. Fit KMeans with optimal clusters (assume K=5 after elbow method)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# 4. Visualize clusters with color coding
plt.figure(figsize=(8,6))
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=80, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=80, c='blue', label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=80, c='green', label='Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=80, c='cyan', label='Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=80, c='magenta', label='Cluster 5')

# Plot cluster centers
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=200, c='yellow', marker='X', label='Centroids')

plt.title("Customer Segments (K-Means Clustering)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1–100)")
plt.legend()
plt.show()

# 5. Evaluate using Silhouette Score
score = silhouette_score(X, y_kmeans)
print(f"Silhouette Score: {score:.3f}")

# Optional: PCA for high-dimensional data (2D visualization)
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)
# kmeans_pca = KMeans(n_clusters=5, random_state=42).fit(X_pca)
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_pca.labels_, cmap='rainbow')
# plt.title("Clusters Visualized in 2D using PCA")
# plt.show()