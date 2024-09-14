import streamlit as st
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, Birch
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Set title
st.title("UMAP & Clustering (K-Means and BIRCH) with Pre-trained UMAP Model")

# Load pre-trained UMAP model and transformed data
umap_transformed_data = joblib.load('umap_transformed_data.pkl')

# Sidebar to select the clustering algorithm
algorithm_choice = st.sidebar.selectbox("Select Clustering Algorithm", ["K-Means", "BIRCH"])

# Sidebar for common clustering parameters
if algorithm_choice == "K-Means":
    st.sidebar.title("K-Means Clustering Parameters")

    # K-Means parameters
    n_clusters_kmeans = st.sidebar.slider("K-Means Number of Clusters", min_value=2, max_value=10, value=3)
    kmeans_init = st.sidebar.selectbox("K-Means init method", options=['k-means++', 'random'])
    max_iter = st.sidebar.slider("K-Means max_iter", min_value=100, max_value=1000, value=300, step=100)

    # Apply K-Means with user-defined parameters
    kmeans = KMeans(n_clusters=n_clusters_kmeans, init=kmeans_init, max_iter=max_iter, random_state=42)
    kmeans_labels = kmeans.fit_predict(umap_transformed_data)

    # Calculate metrics
    silhouette_avg = silhouette_score(umap_transformed_data, kmeans_labels)
    calinski_harabasz_avg = calinski_harabasz_score(umap_transformed_data, kmeans_labels)
    davies_bouldin_avg = davies_bouldin_score(umap_transformed_data, kmeans_labels)

    # Display results
    st.subheader("Clustering Metrics for K-Means")
    st.write(f"Silhouette Score: {silhouette_avg:.2f}")
    st.write(f"Calinski-Harabasz Index: {calinski_harabasz_avg:.2f}")
    st.write(f"Davies-Bouldin Index: {davies_bouldin_avg:.2f}")

    # Visualization
    st.subheader(f"K-Means Clustering Visualization ({n_clusters_kmeans} Clusters)")
    fig, ax = plt.subplots()
    scatter = ax.scatter(umap_transformed_data[:, 0], umap_transformed_data[:, 1], c=kmeans_labels, cmap='viridis', s=50)
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.colorbar(scatter)
    st.pyplot(fig)

elif algorithm_choice == "BIRCH":
    st.sidebar.title("BIRCH Clustering Parameters")

    # BIRCH parameters
    n_clusters_birch = st.sidebar.slider("BIRCH Number of Clusters", min_value=2, max_value=10, value=3)
    branching_factor = st.sidebar.slider("BIRCH Branching Factor", min_value=10, max_value=100, value=50)
    threshold = st.sidebar.slider("BIRCH Threshold", min_value=0.1, max_value=2.0, value=0.5)

    # Apply BIRCH with user-defined parameters
    birch = Birch(n_clusters=n_clusters_birch, branching_factor=branching_factor, threshold=threshold)
    birch_labels = birch.fit_predict(umap_transformed_data)

    # Calculate metrics
    silhouette_avg = silhouette_score(umap_transformed_data, birch_labels)
    calinski_harabasz_avg = calinski_harabasz_score(umap_transformed_data, birch_labels)
    davies_bouldin_avg = davies_bouldin_score(umap_transformed_data, birch_labels)

    # Display results
    st.subheader("Clustering Metrics for BIRCH")
    st.write(f"Silhouette Score: {silhouette_avg:.2f}")
    st.write(f"Calinski-Harabasz Index: {calinski_harabasz_avg:.2f}")
    st.write(f"Davies-Bouldin Index: {davies_bouldin_avg:.2f}")

    # Visualization
    st.subheader(f"BIRCH Clustering Visualization ({n_clusters_birch} Clusters)")
    fig, ax = plt.subplots()
    scatter = ax.scatter(umap_transformed_data[:, 0], umap_transformed_data[:, 1], c=birch_labels, cmap='viridis', s=50)
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.colorbar(scatter)
    st.pyplot(fig)
