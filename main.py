import streamlit as st
import pandas as pd
import umap
import numpy as np
from sklearn.cluster import KMeans, Birch
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from fcmeans import FCM
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

# Algorithm Selection
st.header("Step 4: Using Pre-saved UMAP Embeddings")
algorithm_choice = st.selectbox("Choose the Clustering Algorithm:", ['K-Means', 'BIRCH', 'Fuzzy C-Means'])

# Load UMAP embeddings based on algorithm choice
if algorithm_choice == 'K-Means':
    umap_transformed_data = np.load('umap_embeddings_km.npy')
elif algorithm_choice == 'BIRCH':
    umap_transformed_data = np.load('umap_embeddings_birch.npy')
else:
    umap_transformed_data = np.load('umap_embeddings_fcm.npy')

# Parameters for K-Means, BIRCH, and Fuzzy C-Means
if algorithm_choice == 'K-Means':
    st.subheader("K-Means Parameters")
    n_clusters = st.slider('Number of Clusters for K-Means:', 2, 10, 3)
    init_method = st.selectbox('Initialization Method for K-Means:', ['k-means++', 'random'])
    max_iter = st.slider('Max Iterations for K-Means:', 100, 1000, 300)
    
    # Apply K-Means
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
    st.subheader("BIRCH Parameters")
    n_clusters = st.slider('Number of Clusters for BIRCH:', 2, 10, 2)
    threshold = st.slider('Threshold for BIRCH:', 0.1, 1.0, 0.3)
    branching_factor = st.slider('Branching Factor for BIRCH:', 10, 100, 50)
    
    # Apply BIRCH
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

elif algorithm_choice == 'Fuzzy C-Means':
    st.subheader("Fuzzy C-Means Parameters")
    optimal_num_clusters = st.slider('Number of Clusters for FCM:', 2, 10, 2)
    fuzziness = st.slider('Fuzziness (m):', 1.1, 3.0, 2.0)
    max_iter = st.slider('Max Iterations for FCM:', 100, 1000, 300)
    error = st.slider('Stopping Criterion (Error):', 1e-6, 1e-3, 1e-5, format="%.0e")
    
    # Apply Fuzzy C-Means
    fcm = FCM(n_clusters=optimal_num_clusters, m=fuzziness, max_iter=max_iter, error=error, random_state=42)
    fcm.fit(umap_transformed_data)
    fcm_labels = fcm.predict(umap_transformed_data)
    
    # Evaluation Metrics
    silhouette_avg = silhouette_score(umap_transformed_data, fcm_labels)
    calinski_harabasz_avg = calinski_harabasz_score(umap_transformed_data, fcm_labels)
    davies_bouldin_avg = davies_bouldin_score(umap_transformed_data, fcm_labels)
    
    st.write(f"Silhouette Score: {silhouette_avg:.2f}")
    st.write(f"Calinski-Harabasz Score: {calinski_harabasz_avg:.2f}")
    st.write(f"Davies-Bouldin Score: {davies_bouldin_avg:.2f}")
    
    # Plotting the Clusters for Fuzzy C-Means
    st.subheader("Fuzzy C-Means Cluster Visualization")
    plt.figure(figsize=(8, 6))
    plt.scatter(umap_transformed_data[:, 0], umap_transformed_data[:, 1], c=fcm_labels, cmap='viridis', s=50)
    plt.title(f"Fuzzy C-Means Clustering with {optimal_num_clusters} Clusters\n"
              f"Silhouette Score = {silhouette_avg:.2f}\n"
              f"CH Index = {calinski_harabasz_avg:.2f}\n"
              f"DB Index = {davies_bouldin_avg:.2f}")
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.grid(True)
    st.pyplot(plt)
