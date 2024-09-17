import streamlit as st
import pandas as pd
import umap
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Title
st.title("Clustering Analysis of Global Sustainable Energy Data")

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

# Clustering with UMAP and K-Means
st.header("Step 4: Clustering with UMAP and K-Means")
n_components = st.slider('Select Number of UMAP Components:', 2, 50, 10)
n_clusters = st.slider('Select Number of Clusters:', 2, 10, 3)

# UMAP Transformation
umap_model = umap.UMAP(n_components=n_components, random_state=42)
umap_transformed_data = umap_model.fit_transform(X_scaled)

# K-Means Clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
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

