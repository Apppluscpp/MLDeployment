import streamlit as st
import pandas as pd
import umap
import numpy as np
from sklearn.cluster import KMeans, Birch, AgglomerativeClustering, MeanShift
from fcmeans import FCM
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Title
st.title("Clustering Analysis with Pre-saved UMAP Embeddings")

# Load Dataset
file_path = 'global-data-on-sustainable-energy.csv'
df = pd.read_csv(file_path)

# Data Pre-processing
df.drop(columns=['Financial flows to developing countries (US $)', 
                 'Renewables (% equivalent primary energy)',
                 'Renewable-electricity-generating-capacity-per-capita'], inplace=True)

columns_to_fill_mean = ['Access to clean fuels for cooking', 
                        'Renewable energy share in the total final energy consumption (%)',
                        'Electricity from nuclear (TWh)', 
                        'Energy intensity level of primary energy (MJ/$2017 PPP GDP)',
                        'Value_co2_emissions_kt_by_country', 'gdp_growth', 'gdp_per_capita']
df[columns_to_fill_mean] = df[columns_to_fill_mean].apply(lambda x: x.fillna(x.mean()))

df.dropna(inplace=True)

df.rename(columns={
    "Entity": "Country",
    "gdp_per_capita": "GDP per Capita",
    "Value_co2_emissions_kt_by_country": "CO2 Emissions",
    "Electricity from fossil fuels (TWh)": "Electricity Fossil",
    "Electricity from nuclear (TWh)": "Electricity Nuclear",
    "Electricity from renewables (TWh)": "Electricity Renewables",
    "Renewable energy share in the total final energy consumption (%)": "Renewable Share",
    "Primary energy consumption per capita (kWh/person)": "Energy per Capita",
    "Access to clean fuels for cooking": "Clean Fuels Access",
    "Access to electricity (% of population)": "Electricity Access",
    "Low-carbon electricity (% electricity)": "Low-carbon Electricity",
    "Energy intensity level of primary energy (MJ/$2017 PPP GDP)": "Energy Intensity"
}, inplace=True)

df.rename(columns={'Density\\n(P/Km2)': 'Density'}, inplace=True)
df['Density'] = df['Density'].astype(str).str.replace(',', '').astype(int)

# Feature Selection and Grouping by Country
df.drop(columns=['Year', 'Latitude', 'Longitude', 'Land Area(Km2)'], inplace=True)
grouped_data = df.groupby('Country').mean()

# Scaling
numeric_cols = grouped_data.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(grouped_data[numeric_cols])

# Algorithm Selection and UMAP Embeddings
st.header("Step 4: Using Pre-saved UMAP Embeddings")
algorithm_choice = st.selectbox("Choose the Clustering Algorithm:", ['K-Means', 'BIRCH', 'Hierarchical', 'Mean-Shift', 'Fuzzy C-Means'])

# Default values for UMAP usage check
use_predefined_umap = True  # Assume predefined UMAP unless parameters are changed

# K-Means Parameters
if algorithm_choice == 'K-Means':
    st.subheader("K-Means Parameters")
    n_clusters = st.slider('Number of Clusters for K-Means:', 2, 10, 3)
    init_method = st.selectbox('Initialization Method for K-Means:', ['k-means++', 'random'])
    max_iter = st.slider('Max Iterations for K-Means:', 100, 1000, 300)
    
    # If parameters are changed from default values, recompute UMAP
    if init_method != 'k-means++' or max_iter != 300:
        use_predefined_umap = False
    elif init_method == 'k-means++' and max_iter == 300:
        use_predefined_umap = True  # Reset back to predefined UMAP if defaults are restored

# BIRCH Parameters
elif algorithm_choice == 'BIRCH':
    st.subheader("BIRCH Parameters")
    n_clusters = st.slider('Number of Clusters for BIRCH:', 2, 10, 2)
    threshold = st.slider('Threshold for BIRCH:', 0.1, 1.0, 0.3)
    branching_factor = st.slider('Branching Factor for BIRCH:', 10, 100, 50)
    
    # If parameters are changed from default values, recompute UMAP
    if threshold != 0.1 or branching_factor != 20:
        use_predefined_umap = False
    elif threshold == 0.1 and branching_factor == 20:
        use_predefined_umap = True  # Reset back to predefined UMAP if defaults are restored

# Hierarchical Clustering Parameters
elif algorithm_choice == 'Hierarchical':
    st.subheader("Hierarchical Clustering Parameters")
    n_clusters = st.slider('Number of Clusters for Hierarchical Clustering:', 2, 10, 2)
    linkage_method = st.selectbox('Linkage Method for Hierarchical Clustering:', ['single', 'complete', 'average', 'ward'])

    # If parameters are changed from default values, recompute UMAP
    if linkage_method != 'single':
        use_predefined_umap = False
    elif linkage_method == 'single':
        use_predefined_umap = True  # Reset back to predefined UMAP if defaults are restored

# Mean-Shift Clustering Parameters
elif algorithm_choice == 'Mean-Shift':
    st.subheader("Mean-Shift Parameters")
    optimal_bandwidth = st.slider('Bandwidth for Mean-Shift:', 0.1, 5.0, 2.82)
    
    # If bandwidth is changed from default value, recompute UMAP
    if optimal_bandwidth != 2.82:
        use_predefined_umap = False
    elif optimal_bandwidth == 2.82:
        use_predefined_umap = True  # Reset back to predefined UMAP if defaults are restored

# Fuzzy C-Means Parameters
elif algorithm_choice == 'Fuzzy C-Means':
    st.subheader("Fuzzy C-Means Parameters")
    error = st.slider('Error Tolerance for Fuzzy C-Means:', 1e-6, 1e-3, 1e-5)
    m_value = st.slider('Fuzziness Value (m) for Fuzzy C-Means:', 1.0, 3.0, 1.5)
    max_iter_fcm = st.slider('Max Iterations for Fuzzy C-Means:', 100, 1000, 150)

    # If FCM parameters are changed from default values, recompute UMAP
    if error != 1e-5 or m_value != 1.5 or max_iter_fcm != 150:
        use_predefined_umap = False
    elif error == 1e-5 and m_value == 1.5 and max_iter_fcm == 150:
        use_predefined_umap = True  # Reset back to predefined UMAP if defaults are restored

# Use pre-saved UMAP embeddings only if all parameters except number of clusters are unchanged
if use_predefined_umap:
    if algorithm_choice == 'K-Means':
        umap_transformed_data = np.load('umap_embeddings_km.npy')
    elif algorithm_choice == 'BIRCH':
        umap_transformed_data = np.load('umap_embeddings_birch.npy')
    elif algorithm_choice == 'Hierarchical':
        umap_transformed_data = np.load('umap_embeddings_hc.npy')
    elif algorithm_choice == 'Mean-Shift':
        umap_transformed_data = np.load('umap_embeddings_ms.npy')
    elif algorithm_choice == 'Fuzzy C-Means':
        umap_transformed_data = np.load('umap_embeddings_fcm.npy')
else:
    # Recompute UMAP embeddings
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    umap_transformed_data = umap_model.fit_transform(X_scaled)

# Apply Clustering based on user-selected algorithm
if algorithm_choice == 'K-Means':
    kmeans = KMeans(n_clusters=n_clusters, init=init_method, max_iter=max_iter, random_state=42)
    kmeans_labels = kmeans.fit_predict(umap_transformed_data)
    
    # Evaluation Metrics
    silhouette_avg = silhouette_score(umap_transformed_data, kmeans_labels)
    calinski_harabasz_avg = calinski_harabasz_score(umap_transformed_data, kmeans_labels)
    davies_bouldin_avg = davies_bouldin_score(umap_transformed_data, kmeans_labels)
    
    st.write(f"Silhouette Score: {silhouette_avg:.2f}")
    st.write(f"Calinski-Harabasz Score: {calinski_harabasz_avg:.2f}")
    st.write(f"Davies-Bouldin Score: {davies_bouldin_avg:.2f}")
    
    # Plotting the Clusters
    st.subheader("K-Means Cluster Visualization")
    plt.figure(figsize=(8, 6))
    plt.scatter(umap_transformed_data[:, 0], umap_transformed_data[:, 1], c=kmeans_labels, cmap='viridis', s=50)
    plt.title(f"K-Means Clustering with {n_clusters} Clusters")
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    st.pyplot(plt)

elif algorithm_choice == 'BIRCH':
    birch = Birch(n_clusters=n_clusters, threshold=threshold, branching_factor=branching_factor)
    birch_labels = birch.fit_predict(umap_transformed_data)
    
    # Evaluation Metrics
    silhouette_avg = silhouette_score(umap_transformed_data, birch_labels)
    calinski_harabasz_avg = calinski_harabasz_score(umap_transformed_data, birch_labels)
    davies_bouldin_avg = davies_bouldin_score(umap_transformed_data, birch_labels)
    
    st.write(f"Silhouette Score: {silhouette_avg:.2f}")
    st.write(f"Calinski-Harabasz Score: {calinski_harabasz_avg:.2f}")
    st.write(f"Davies-Bouldin Score: {davies_bouldin_avg:.2f}")
    
    # Plotting the Clusters
    st.subheader("BIRCH Cluster Visualization")
    plt.figure(figsize=(8, 6))
    plt.scatter(umap_transformed_data[:, 0], umap_transformed_data[:, 1], c=birch_labels, cmap='viridis', s=50)
    plt.title(f"BIRCH Clustering with Best Parameters\n"
              f"Silhouette Score = {silhouette_avg:.2f}")
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    st.pyplot(plt)

elif algorithm_choice == 'Hierarchical':
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    hierarchical_labels = hierarchical.fit_predict(umap_transformed_data)
    
    # Evaluation Metrics
    silhouette_avg = silhouette_score(umap_transformed_data, hierarchical_labels)
    calinski_harabasz_avg = calinski_harabasz_score(umap_transformed_data, hierarchical_labels)
    davies_bouldin_avg = davies_bouldin_score(umap_transformed_data, hierarchical_labels)
    
    st.write(f"Silhouette Score: {silhouette_avg:.2f}")
    st.write(f"Calinski-Harabasz Score: {calinski_harabasz_avg:.2f}")
    st.write(f"Davies-Bouldin Score: {davies_bouldin_avg:.2f}")
    
    # Plotting the Clusters
    st.subheader("Hierarchical Clustering Visualization")
    plt.figure(figsize=(8, 6))
    plt.scatter(umap_transformed_data[:, 0], umap_transformed_data[:, 1], c=hierarchical_labels, cmap='viridis', s=50)
    plt.title(f"Hierarchical Clustering with {n_clusters} Clusters\n"
              f"Silhouette Score = {silhouette_avg:.2f}")
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    st.pyplot(plt)

elif algorithm_choice == 'Mean-Shift':
    mean_shift = MeanShift(bandwidth=optimal_bandwidth)
    mean_shift_labels = mean_shift.fit_predict(umap_transformed_data)
    
    # Number of clusters found by Mean-Shift
    n_clusters = len(np.unique(mean_shift_labels))
    
    # Evaluation Metrics
    if n_clusters > 1:  # Ensure there are multiple clusters
        silhouette_ms = silhouette_score(umap_transformed_data, mean_shift_labels)
        calinski_harabasz_ms = calinski_harabasz_score(umap_transformed_data, mean_shift_labels)
        davies_bouldin_ms = davies_bouldin_score(umap_transformed_data, mean_shift_labels)

        st.write(f"Silhouette Score: {silhouette_ms:.2f}")
        st.write(f"Calinski-Harabasz Index: {calinski_harabasz_ms:.2f}")
        st.write(f"Davies-Bouldin Index: {davies_bouldin_ms:.2f}")
        
        # Plot the clustering results
        st.subheader("Mean-Shift Cluster Visualization")
        plt.figure(figsize=(8, 6))
        plt.scatter(umap_transformed_data[:, 0], umap_transformed_data[:, 1], c=mean_shift_labels, cmap='viridis', s=50)
        plt.title(f"Mean-Shift Clustering with Bandwidth = {optimal_bandwidth}\n"
                  f"Silhouette Score = {silhouette_ms:.2f}, CH Index = {calinski_harabasz_ms:.2f}, DB Index = {davies_bouldin_ms:.2f}")
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        st.pyplot(plt)
    else:
        st.write("Only one cluster found, no further metrics calculated.")

elif algorithm_choice == 'Fuzzy C-Means':
    fcm = FCM(n_clusters=optimal_num_clusters, m=m_value, max_iter=max_iter_fcm, error=error, random_state=42)
    fcm.fit(umap_transformed_data)
    fcm_labels = fcm.predict(umap_transformed_data)
    
    # Evaluation Metrics
    silhouette_avg = silhouette_score(umap_transformed_data, fcm_labels)
    calinski_harabasz_avg = calinski_harabasz_score(umap_transformed_data, fcm_labels)
    davies_bouldin_avg = davies_bouldin_score(umap_transformed_data, fcm_labels)
    
    st.write(f"Silhouette Score: {silhouette_avg:.2f}")
    st.write(f"Calinski-Harabasz Score: {calinski_harabasz_avg:.2f}")
    st.write(f"Davies-Bouldin Score: {davies_bouldin_avg:.2f}")
    
    # Plotting the Clusters
    st.subheader("Fuzzy C-Means Cluster Visualization")
    plt.figure(figsize=(8, 6))
    plt.scatter(umap_transformed_data[:, 0], umap_transformed_data[:, 1], c=fcm_labels, cmap='viridis', s=50)
    plt.title(f"Fuzzy C-Means Clustering with {optimal_num_clusters} Clusters\n"
              f"Silhouette Score = {silhouette_avg:.2f}")
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    st.pyplot(plt)
