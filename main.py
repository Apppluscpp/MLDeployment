import streamlit as st
import pandas as pd
import umap
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Title
st.title("Clustering Analysis of Global Sustainable Energy Data with Pre-saved UMAP Embeddings")

# Load Dataset
st.header("Step 1: Load Dataset")
file_path = 'global-data-on-sustainable-energy.csv'
df = pd.read_csv(file_path)
st.write("Dataset Loaded Successfully!")
st.write(df.head())

# Data Pre-processing
st.header("Step 2: Data Pre-processing")
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
st.header("Step 3: Feature Scaling")
numeric_cols = grouped_data.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(grouped_data[numeric_cols])
st.write("Data Scaled Successfully!")

# Load UMAP embeddings if available
use_saved_embeddings = st.checkbox("Use Pre-saved UMAP Embeddings")
if use_saved_embeddings:
    umap_transformed_data = np.load('umap_embeddings_km.npy')
    st.write("Loaded Pre-saved UMAP Embeddings")
else:
    # UMAP parameters
    st.subheader("UMAP Parameters")
    n_components = st.slider('Number of UMAP Components:', 2, 50, 10)
    n_neighbors = st.slider('Number of UMAP Neighbors:', 5, 50, 15)
    min_dist = st.slider('Minimum Distance for UMAP:', 0.0, 1.0, 0.1)

    # Apply UMAP
    umap_model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    umap_transformed_data = umap_model.fit_transform(X_scaled)
    st.write(f"UMAP applied with {n_components} components, {n_neighbors} neighbors, and {min_dist} min distance")

# K-Means parameters
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
st.subheader("Cluster Visualization")
plt.figure(figsize=(8, 6))
plt.scatter(umap_transformed_data[:, 0], umap_transformed_data[:, 1], c=kmeans_labels, cmap='viridis', s=50)
plt.title(f"K-Means Clustering with {n_clusters} Clusters")
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
st.pyplot(plt)
