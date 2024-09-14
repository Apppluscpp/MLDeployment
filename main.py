import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Set title
st.title("UMAP & K-Means Clustering App")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
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

    # Sidebar for UMAP and K-Means parameters
    st.sidebar.title("Clustering Parameters")
    n_components = st.sidebar.slider("Number of UMAP Components", min_value=2, max_value=50, value=10)
    n_clusters = st.sidebar.slider("Number of K-Means Clusters", min_value=2, max_value=10, value=3)
    
    # Apply UMAP
    umap_model = umap.UMAP(n_components=n_components, random_state=42)
    umap_transformed_data = umap_model.fit_transform(X_scaled)

    # Apply K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(umap_transformed_data)

    # Calculate metrics
    silhouette_avg = silhouette_score(umap_transformed_data, kmeans_labels)
    calinski_harabasz_avg = calinski_harabasz_score(umap_transformed_data, kmeans_labels)
    davies_bouldin_avg = davies_bouldin_score(umap_transformed_data, kmeans_labels)

    # Display results
    st.subheader("Clustering Metrics")
    st.write(f"Silhouette Score: {silhouette_avg:.2f}")
    st.write(f"Calinski-Harabasz Index: {calinski_harabasz_avg:.2f}")
    st.write(f"Davies-Bouldin Index: {davies_bouldin_avg:.2f}")

    # Visualization
    st.subheader("Clustering Visualization")
    fig, ax = plt.subplots()
    scatter = ax.scatter(umap_transformed_data[:, 0], umap_transformed_data[:, 1], c=kmeans_labels, cmap='viridis', s=50)
    plt.title(f"UMAP with {n_components} Components and K-Means Clustering ({n_clusters} Clusters)")
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.colorbar(scatter)
    st.pyplot(fig)
