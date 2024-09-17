

import pandas as pd
import streamlit as st
import umap
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load dataset
file_path = 'global-data-on-sustainable-energy.csv'
df = pd.read_csv(file_path)

# Data Pre-processing & Feature Selection
df.drop(columns=['Financial flows to developing countries (US $)', 
                 'Renewables (% equivalent primary energy)', 
                 'Renewable-electricity-generating-capacity-per-capita'], inplace=True)

columns_to_fill_mean = ['Access to clean fuels for cooking', 
                        'Renewable energy share in the total final energy consumption (%)', 
                        'Electricity from nuclear (TWh)', 
                        'Energy intensity level of primary energy (MJ/$2017 PPP GDP)', 
                        'Value_co2_emissions_kt_by_country', 
                        'gdp_growth', 
                        'gdp_per_capita']
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
df.drop(columns=['Latitude', 'Longitude', 'Land Area(Km2)'], inplace=True)

# Drop Year and group by Country
df.drop(columns=['Year'], inplace=True)
grouped_data = df.groupby('Country').mean()

# Scale data
X_scaled = StandardScaler().fit_transform(grouped_data)

