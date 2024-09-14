import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score

# Set title for the app
st.title("UMAP & BIRCH Clustering App")

# Load dataset
file_path = 'global-data-on-sustainable-energy.csv'
df = pd.read_csv(file_path)
st.write("Dataset loaded successfully!")

# Data Preprocessing
df.drop(columns=['Financial flows to developing countries (US $)', 'Renewables (% equivalent primary energy)', 
                 'Renewable-electricity-generating-capacity-per-capita'], inplace=True)

# Fill missing values with mean
columns_to_fill_mean = ['Access to clean fuels for cooking', 'Renewable energy share in the total final energy consumption (%)',
                        'Electricity from nuclear (TWh)', 'Energy intensity level of primary energy (MJ/$2017 PPP GDP)',
                        'Value_co2_emissions_kt_by_country', 'gdp_growth', 'gdp_per_capita']
df[columns_to_fill_mean] = df[columns_to_fill_mean].apply(lambda x: x.fillna(x.mean()))

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

df.rename(columns={'Density\\n(P/Km2)': 'Density'}, inplace=True)
df['Density'] = df['Density'].astype(str).str.replace(',', '').astype(int)

# Drop columns not useful for clustering
df.drop(columns=['Latitude', 'Longitude', 'Land Area(Km2)', 'Year'], inplace=True)

# Group data by country and scale
grouped_data = df.groupby('Country').mean()
numeric_cols = grouped_data.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(grouped_data[numeric_cols])

# Sidebar for UMAP and BIRCH parameters
st.sidebar.title("Clustering Parameters")

# UMAP parameters
n_neighbors = st.sidebar.slider("UMAP n_neighbors", min_value=2, max_value=50, value=15)
min_dist = st.sidebar.slider("UMAP min_dist", min_value=0.0, max_value=1.0, value=0.1)
n_components = st.sidebar.slider("UMAP n_components", min_value=2, max_value=50, value=10)

# BIRCH parameters
branching_factor = st.sidebar.slider("BIRCH Branching Factor", min_value=20, max_value=100, value=50, step=10)
threshold = st.sidebar.slider("BIRCH Threshold", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
n_clusters = st.sidebar.slider("BIRCH Number of Clusters", min_value=2, max_value=10, value=3)

# Apply UMAP with user-defined parameters
umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=42)
umap_transformed_data = umap_model.fit_transform(X_scaled)

# Apply BIRCH with user-defined parameters
birch = Birch(n_clusters=n_clusters, branching_factor=branching_factor, threshold=threshold)
birch_labels = birch.fit_predict(umap_transformed_data)

# Calculate silhouette score
silhouette_avg = silhouette_score(umap_transformed_data, birch_labels)

# Display results
st.subheader(f"Clustering Metrics")
st.write(f"Silhouette Score: {silhouette_avg:.2f}")

# Visualization
st.subheader("Clustering Visualization")
fig, ax = plt.subplots()
scatter = ax.scatter(umap_transformed_data[:, 0], umap_transformed_data[:, 1], c=birch_labels, cmap='viridis', s=50)
plt.title(f"UMAP with {n_components} Components and BIRCH Clustering ({n_clusters} Clusters)")
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.colorbar(scatter)
st.pyplot(fig)
