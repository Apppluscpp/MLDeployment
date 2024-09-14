import numpy as np
import streamlit as st
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score

# Load precomputed UMAP embeddings
umap_transformed_data = np.load('umap_embeddings.npy')

# Sidebar for BIRCH parameters
branching_factor = st.sidebar.slider("BIRCH Branching Factor", min_value=20, max_value=100, value=50)
threshold = st.sidebar.slider("BIRCH Threshold", min_value=0.1, max_value=1.0, value=0.5)
n_clusters = st.sidebar.slider("BIRCH Number of Clusters", min_value=2, max_value=10, value=3)

# Apply BIRCH with user-defined parameters
birch = Birch(n_clusters=n_clusters, branching_factor=branching_factor, threshold=threshold)
birch_labels = birch.fit_predict(umap_transformed_data)

# Calculate silhouette score
silhouette_avg = silhouette_score(umap_transformed_data, birch_labels)
st.write(f"Silhouette Score: {silhouette_avg:.2f}")

# Visualization code
