#===========================================================
# Importation des librairies
#===========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import plotly.express as px
import time
import random
random.seed(123)


#===========================================================
# Chargement des données
#===========================================================

#Chargement de financial
financial = pd.read_csv('data/financial_clean.csv')

#==========================================================

# Titre de l'application
st.title("Géolocalisation des transactions")

# Charger les données
csv_file_path = "data/individual_data_b.csv"
individual_data = pd.read_csv(csv_file_path)

# Conversion de la colonne 'date' en format datetime
individual_data["date"] = pd.to_datetime(individual_data["date"])

# Création de deux colonnes pour les informations
info_col1, info_col2 = st.columns(2)

# Colonne 1 : Saisie du numéro de carte
with info_col1:
    st.subheader("Saisissez les informations")
    cc_num = st.text_input("Numéro de carte bancaire", value = "2703186189652095")
    filtered_data = pd.DataFrame()  # Initialisation d'un DataFrame vide

    if cc_num:
        filtered_data = individual_data[individual_data["cc_num"].astype(str) == cc_num]
        if filtered_data.empty:
            st.warning("Aucune transaction trouvée pour ce numéro de carte.")
        else:
            st.success(f"{len(filtered_data)} transactions trouvées pour la carte {cc_num}.")

# Colonne 2 : Choix de couleur
with info_col2:
    st.subheader("Choisissez une couleur pour les points de la carte")
    color_points = st.color_picker("Couleur :", "#FF0000")

# Filtrage par date avec un slider
if not filtered_data.empty:
    st.sidebar.header("Filtres par date")
    
    # Définir la plage de dates disponible et convertir en datetime.date
    min_date = filtered_data["date"].min().date()
    max_date = filtered_data["date"].max().date()
    
    # Slider pour sélectionner la plage de dates
    date_range = st.sidebar.slider(
        "Sélectionnez une plage de dates",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )
    
    # Filtrer les données par la plage de dates sélectionnée
    filtered_data = filtered_data[
        (filtered_data["date"] >= pd.Timestamp(date_range[0])) & 
        (filtered_data["date"] <= pd.Timestamp(date_range[1]))
    ]
    
    if filtered_data.empty:
        st.warning("Aucune transaction trouvée pour la plage de dates sélectionnée.")

    else:
            # Mettre à jour dynamiquement le nombre de transactions après filtrage
            st.success(f"{len(filtered_data)} transactions trouvées après filtrage par date.")
            
# Affichage des visualisations si des transactions sont trouvées
if not filtered_data.empty:
    # Création de deux colonnes pour la carte et le graphique
    map_col, chart_col = st.columns(2)

    # Colonne 1 : Carte interactive avec Plotly
    with map_col:
        st.subheader("Carte des transactions")
        map_data= individual_data.head(200)
        fig = px.scatter_mapbox(
            map_data,
            lat="lat",
            lon="long",
            hover_name="city",
            hover_data={
                "daily_amount$": True,
                "date": True,
                "category": True,
                "is_fraud":True,
                "daily_fraud_amount$":True,
                "merchant":True
               
            },
            color_discrete_sequence=[color_points],
            title="Carte des transactions",
            zoom=4,
        )
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)

    # Colonne 2 : Graphique des transactions
    with chart_col:
        st.subheader("Transactions par jour ou par période")
        daily_data = filtered_data.groupby(filtered_data["date"].dt.date).size().reset_index(name="count")
        fig = px.bar(
            daily_data,
            x="date",
            y="count",
            title="Nombre de transactions ",
            color_discrete_sequence=[color_points]
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Les résultats s'afficheront ici une fois qu'un numéro de carte et une période seront saisis.")

