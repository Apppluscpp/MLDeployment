import streamlit as st
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Set title
st.title("UMAP & K-Means Clustering with Pre-trained UMAP Model")

# Load pre-trained UMAP model and transformed data
umap_transformed_data = joblib.load('umap_transformed_data.pkl')

# Sidebar for K-Means parameters
st.sidebar.title("K-Means Clustering Parameters")

# K-Means parameters
n_clusters = st.sidebar.slider("K-Means Number of Clusters", min_value=2, max_value=10, value=3)
kmeans_init = st.sidebar.selectbox("K-Means init method", options=['k-means++', 'random'])
max_iter = st.sidebar.slider("K-Means max_iter", min_value=100, max_value=1000, value=300, step=100)

# Apply K-Means with user-defined parameters
kmeans = KMeans(n_clusters=n_clusters, init=kmeans_init, max_iter=max_iter, random_state=42)
kmeans_labels = kmeans.fit_predict(umap_transformed_data)

# Calculate metrics
silhouette_avg = silhouette_score(umap_transformed_data, kmeans_labels)
calinski_harabasz_avg = calinski_harabasz_score(umap_transformed_data, kmeans_labels)
davies_bouldin_avg = davies_bouldin_score(umap_transformed_data, kmeans_labels)

# Display results
st.subheader("Clustering Metrics")
st.write(f"Silhouette Score: {silhouette_avg:.2f}")
st.write(f"Calinski-Harabasz Index: {calinski_harabasz_avg:.2f}")
st.write(f"Davies-Bouldin Index: {davies_bouldin_avg:.2f}")

# Visualization
st.subheader("Clustering Visualization")
fig, ax = plt.subplots()
scatter = ax.scatter(umap_transformed_data[:, 0], umap_transformed_data[:, 1], c=kmeans_labels, cmap='viridis', s=50)
plt.title(f"Pre-trained UMAP with K-Means Clustering ({n_clusters} Clusters)")
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.colorbar(scatter)
st.pyplot(fig)
