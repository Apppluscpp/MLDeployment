import streamlit as st
import pandas as pd
import umap
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt

# Title of the Streamlit App
st.title('Global Sustainable Energy Data Clustering with UMAP & K-Means')

# 1. Load Dataset
st.header('Load Dataset')
file_path = 'global-data-on-sustainable-energy.csv'
df = pd.read_csv(file_path)
st.write("Dataset preview:")
st.write(df.head())

# 2. Data Pre-processing
st.header('Data Pre-processing')
df.drop(columns=['Financial flows to developing countries (US $)', 'Renewables (% equivalent primary energy)',
                 'Renewable-electricity-generating-capacity-per-capita'], inplace=True)

columns_to_fill_mean = ['Access to clean fuels for cooking', 'Renewable energy share in the total final energy consumption (%)',
                        'Electricity from nuclear (TWh)', 'Energy intensity level of primary energy (MJ/$2017 PPP GDP)',
                        'Value_co2_emissions_kt_by_country', 'gdp_growth', 'gdp_per_capita']
df[columns_to_fill_mean] = df[columns_to_fill_mean].apply(lambda x: x.fillna(x.mean()))
df.dropna(inplace=True)

df.rename(columns={
    "Entity": "Country", "gdp_per_capita": "GDP per Capita", "Value_co2_emissions_kt_by_country": "CO2 Emissions",
    "Electricity from fossil fuels (TWh)": "Electricity Fossil", "Electricity from nuclear (TWh)": "Electricity Nuclear",
    "Electricity from renewables (TWh)": "Electricity Renewables", "Renewable energy share in the total final energy consumption (%)": "Renewable Share",
    "Primary energy consumption per capita (kWh/person)": "Energy per Capita", "Access to clean fuels for cooking": "Clean Fuels Access",
    "Access to electricity (% of population)": "Electricity Access", "Low-carbon electricity (% electricity)": "Low-carbon Electricity",
    "Energy intensity level of primary energy (MJ/$2017 PPP GDP)": "Energy Intensity"
}, inplace=True)
df.rename(columns={'Density\\n(P/Km2)': 'Density'}, inplace=True)
df['Density'] = df['Density'].astype(str).str.replace(',', '').astype(int)
df.drop(columns=['Latitude', 'Longitude', 'Land Area(Km2)', 'Year'], inplace=True)

# 3. Scaling Numerical Data
st.header('Scaling Data')
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[numeric_cols])

# 4. Determine the best n_components with UMAP
st.header('Determine Best n_components for UMAP')
n_components_list = [2, 5, 10, 20, 30, 50]
silhouette_scores = []

for n in n_components_list:
    umap_model = umap.UMAP(n_components=n, random_state=42)
    umap_transformed_data = umap_model.fit_transform(X_scaled)
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_labels = kmeans.fit_predict(umap_transformed_data)
    silhouette_avg = silhouette_score(umap_transformed_data, kmeans_labels)
    silhouette_scores.append(silhouette_avg)

best_n_components = n_components_list[np.argmax(silhouette_scores)]
st.write(f"The best n_components for UMAP is: {best_n_components} with a Silhouette Score of {max(silhouette_scores)}")

# Plot Silhouette Scores
fig, ax = plt.subplots()
ax.plot(n_components_list, silhouette_scores, marker='o', color='b')
ax.set_title('Silhouette Score vs UMAP n_components')
ax.set_xlabel('n_components')
ax.set_ylabel('Silhouette Score')
st.pyplot(fig)

# 5. Perform clustering with the best n_components
st.header('Clustering with Best Parameters')
umap_model = umap.UMAP(n_neighbors=5, min_dist=0.0, n_components=best_n_components, random_state=42)
umap_transformed_data = umap_model.fit_transform(X_scaled)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(umap_transformed_data)

silhouette_avg = silhouette_score(umap_transformed_data, kmeans_labels)
calinski_harabasz_avg = calinski_harabasz_score(umap_transformed_data, kmeans_labels)
davies_bouldin_avg = davies_bouldin_score(umap_transformed_data, kmeans_labels)

st.write(f"Silhouette Score: {silhouette_avg}")
st.write(f"Calinski-Harabasz Index: {calinski_harabasz_avg}")
st.write(f"Davies-Bouldin Index: {davies_bouldin_avg}")

# Final clustering plot
fig, ax = plt.subplots()
scatter = ax.scatter(umap_transformed_data[:, 0], umap_transformed_data[:, 1], c=kmeans_labels, cmap='viridis')
ax.set_title(f"K-Means with {3} Clusters")
ax.set_xlabel('UMAP1')
ax.set_ylabel('UMAP2')
st.pyplot(fig)
