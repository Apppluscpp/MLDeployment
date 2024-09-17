import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, Birch
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Set title
st.title("UMAP, K-Means & BIRCH Clustering App")

# Loading the dataset from the file path
file_path = 'global-data-on-sustainable-energy.csv'
df = pd.read_csv(file_path)
st.write("Dataset loaded successfully!")

# Data Preprocessing
# Drop columns with high number of missing values
df.drop(columns=['Financial flows to developing countries (US $)', 'Renewables (% equivalent primary energy)', 
                 'Renewable-electricity-generating-capacity-per-capita'], inplace=True)

# Fill missing values with mean
columns_to_fill_mean = ['Access to clean fuels for cooking', 'Renewable energy share in the total final energy consumption (%)',
                        'Electricity from nuclear (TWh)', 'Energy intensity level of primary energy (MJ/$2017 PPP GDP)',
                        'Value_co2_emissions_kt_by_country', 'gdp_growth', 'gdp_per_capita']
df[columns_to_fill_mean] = df[columns_to_fill_mean].apply(lambda x: x.fillna(x.mean()))

# Drop remaining rows with missing values
df = df.dropna()

# Rename columns for simplicity
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

# Convert 'Density' column and filter numerical data
df.rename(columns={'Density\\n(P/Km2)': 'Density'}, inplace=True)
df['Density'] = df['Density'].astype(str).str.replace(',', '').astype(int)

# Drop columns not useful for clustering
df.drop(columns=['Latitude', 'Longitude', 'Land Area(Km2)', 'Year'], inplace=True)

# Group data by country
grouped_data = df.groupby('Country').mean()

# Scale the numerical data
numeric_cols = grouped_data.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(grouped_data[numeric_cols])

# Sidebar for UMAP, K-Means, and BIRCH parameters
st.sidebar.title("Clustering Parameters")

# UMAP parameters
n_neighbors = st.sidebar.slider("UMAP n_neighbors", min_value=2, max_value=50, value=15)
min_dist = st.sidebar.slider("UMAP min_dist", min_value=0.0, max_value=1.0, value=0.1)
n_components = st.sidebar.slider("UMAP n_components", min_value=2, max_value=50, value=10)

# K-Means parameters
n_clusters_kmeans = st.sidebar.slider("K-Means Number of Clusters", min_value=2, max_value=10, value=3)
kmeans_init = st.sidebar.selectbox("K-Means init method", options=['k-means++', 'random'])
max_iter = st.sidebar.slider("K-Means max_iter", min_value=100, max_value=1000, value=300, step=100)

# BIRCH parameters
st.sidebar.subheader("BIRCH Parameters")
n_clusters_birch = st.sidebar.slider("BIRCH Number of Clusters", min_value=2, max_value=10, value=2)
threshold = st.sidebar.slider("BIRCH Threshold", min_value=0.1, max_value=1.0, value=0.5)
branching_factor = st.sidebar.slider("BIRCH Branching Factor", min_value=20, max_value=100, value=50)

# Apply UMAP with user-defined parameters
umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=42)
umap_transformed_data = umap_model.fit_transform(X_scaled)

# Apply K-Means with user-defined parameters
kmeans = KMeans(n_clusters=n_clusters_kmeans, init=kmeans_init, max_iter=max_iter, random_state=42)
kmeans_labels = kmeans.fit_predict(umap_transformed_data)

# Apply BIRCH with user-defined parameters
birch = Birch(n_clusters=n_clusters_birch, threshold=threshold, branching_factor=branching_factor)
birch_labels = birch.fit_predict(umap_transformed_data)

# Calculate metrics for K-Means
silhouette_avg_kmeans = silhouette_score(umap_transformed_data, kmeans_labels)
calinski_harabasz_avg_kmeans = calinski_harabasz_score(umap_transformed_data, kmeans_labels)
davies_bouldin_avg_kmeans = davies_bouldin_score(umap_transformed_data, kmeans_labels)

# Calculate metrics for BIRCH
silhouette_avg_birch = silhouette_score(umap_transformed_data, birch_labels)
calinski_harabasz_avg_birch = calinski_harabasz_score(umap_transformed_data, birch_labels)
davies_bouldin_avg_birch = davies_bouldin_score(umap_transformed_data, birch_labels)

# Display results for K-Means
st.subheader("K-Means Clustering Metrics")
st.write(f"Silhouette Score (K-Means): {silhouette_avg_kmeans:.2f}")
st.write(f"Calinski-Harabasz Index (K-Means): {calinski_harabasz_avg_kmeans:.2f}")
st.write(f"Davies-Bouldin Index (K-Means): {davies_bouldin_avg_kmeans:.2f}")

# Display results for BIRCH
st.subheader("BIRCH Clustering Metrics")
st.write(f"Silhouette Score (BIRCH): {silhouette_avg_birch:.2f}")
st.write(f"Calinski-Harabasz Index (BIRCH): {calinski_harabasz_avg_birch:.2f}")
st.write(f"Davies-Bouldin Index (BIRCH): {davies_bouldin_avg_birch:.2f}")

# Visualization for K-Means
st.subheader("K-Means Clustering Visualization")
fig_kmeans, ax_kmeans = plt.subplots()
scatter_kmeans = ax_kmeans.scatter(umap_transformed_data[:, 0], umap_transformed_data[:, 1], c=kmeans_labels, cmap='viridis', s=50)
plt.title(f"UMAP with {n_components} Components and K-Means Clustering ({n_clusters_kmeans} Clusters)")
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.colorbar(scatter_kmeans)
st.pyplot(fig_kmeans)

# Visualization for BIRCH
st.subheader("BIRCH Clustering Visualization")
fig_birch, ax_birch = plt.subplots()
scatter_birch = ax_birch.scatter(umap_transformed_data[:, 0], umap_transformed_data[:, 1], c=birch_labels, cmap='plasma', s=50)
plt.title(f"UMAP with {n_components} Components and BIRCH Clustering ({n_clusters_birch} Clusters)")
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.colorbar(scatter_birch)
st.pyplot(fig_birch)
