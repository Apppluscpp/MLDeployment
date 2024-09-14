import streamlit as st
import pandas as pd
import umap
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid

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

# 5. Determine the best number of clusters
st.header('Determine Best Number of Clusters')
umap_model = umap.UMAP(n_components=best_n_components, random_state=42)
umap_transformed_data = umap_model.fit_transform(X_scaled)
num_clusters_list = range(2, 11)
silhouette_scores = []
inertia_list = []

for k in num_clusters_list:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans_labels = kmeans.fit_predict(umap_transformed_data)
    inertia = kmeans.inertia_
    inertia_list.append(inertia)
    silhouette_avg = silhouette_score(umap_transformed_data, kmeans_labels)
    silhouette_scores.append(silhouette_avg)

optimal_num_clusters = num_clusters_list[np.argmax(silhouette_scores)]
st.write(f"The optimal number of clusters is: {optimal_num_clusters}")

# 6. Perform Clustering
st.header('Perform Clustering')
umap_model = umap.UMAP(n_components=optimal_num_clusters, random_state=42)
umap_transformed_data = umap_model.fit_transform(X_scaled)
kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=42)
labels = kmeans.fit_predict(umap_transformed_data)
silhouette_avg = silhouette_score(umap_transformed_data, labels)
calinski_harabasz_avg = calinski_harabasz_score(umap_transformed_data, labels)
davies_bouldin_avg = davies_bouldin_score(umap_transformed_data, labels)

st.write(f"Silhouette Score: {silhouette_avg}")
st.write(f"Calinski-Harabasz Index: {calinski_harabasz_avg}")
st.write(f"Davies-Bouldin Index: {davies_bouldin_avg}")

# Final clustering plot
fig, ax = plt.subplots()
scatter = ax.scatter(umap_transformed_data[:, 0], umap_transformed_data[:, 1], c=labels, cmap='viridis')
ax.set_title(f"K-Means with {optimal_num_clusters} Clusters")
ax.set_xlabel('UMAP1')
ax.set_ylabel('UMAP2')
st.pyplot(fig)

# 7. Fine-Tune UMAP Parameters
st.header('Fine-Tune UMAP Parameters')
n_neighbors_list = [5, 10, 15, 30]
min_dist_list = [0.0, 0.1, 0.5]
silhouette_scores_umap = {}

for n_neighbors in n_neighbors_list:
    for min_dist in min_dist_list:
        umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=optimal_num_clusters, random_state=42)
        umap_transformed_data = umap_model.fit_transform(X_scaled)
        kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(umap_transformed_data)
        silhouette_avg = silhouette_score(umap_transformed_data, kmeans_labels)
        silhouette_scores_umap[(n_neighbors, min_dist)] = silhouette_avg

best_params = max(silhouette_scores_umap, key=silhouette_scores_umap.get)
st.write(f"The best UMAP parameters are: n_neighbors = {best_params[0]}, min_dist = {best_params[1]}")

# 8. Fine-Tune K-Means Clustering
st.header('Fine-Tune K-Means Clustering')
param_grid = {'init': ['k-means++', 'random'], 'max_iter': [300, 500, 1000]}
results = []

for params in ParameterGrid(param_grid):
    kmeans = KMeans(n_clusters=optimal_num_clusters, init=params['init'], max_iter=params['max_iter'], random_state=42)
    kmeans_labels = kmeans.fit_predict(umap_transformed_data)
    silhouette_avg = silhouette_score(umap_transformed_data, kmeans_labels)
    calinski_harabasz_avg = calinski_harabasz_score(umap_transformed_data, kmeans_labels)
    davies_bouldin_avg = davies_bouldin_score(umap_transformed_data, kmeans_labels)
    results.append({'params': params, 'silhouette_score': silhouette_avg, 'calinski_harabasz_score': calinski_harabasz_avg,
                    'davies_bouldin_score': davies_bouldin_avg})

results_df = pd.DataFrame(results)
best_result = results_df.loc[results_df['silhouette_score'].idxmax()]
st.write(f"Best Parameters based on Silhouette Score: {best_result}")
