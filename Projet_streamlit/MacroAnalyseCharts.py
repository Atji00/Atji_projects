############################################ LIBRARIES ############################################
import streamlit as st
import pandas as pd

############################################ FONCTIONS ############################################
# Fonction pour filtrer les données par plage de dates avec slider
def filter_by_date_with_slider(data):
    if data.empty:
        return data, None  
    
    st.sidebar.header("Selectionner la date :")

    # Conversion de la date au format datetime (pour éviter soucis de compatibilité)
    filtered_data["date"] = pd.to_datetime(filtered_data["date"], errors="coerce")

    # Définir la plage de dates disponible
    min_date = data["date"].min().date()
    max_date = data["date"].max().date()

    # Ajouter un slider pour sélectionner la plage de dates
    # st.sidebar.header("Filtres par date")
    date_of_interest = st.sidebar.slider(
        "Sélectionnez une plage de dates",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )
    

    # Filtrer les données en fonction de la plage de dates sélectionnée
    df = data[
        (data["date"] >= pd.Timestamp(date_of_interest[0])) &
        (data["date"] <= pd.Timestamp(date_of_interest[1]))
    ]
    
    # Calcul du total de paiements frauduleux
    total_fraud_sum = len(df)
    st.sidebar.caption(f"{total_fraud_sum} résultats")

    # Somme de fraudes totales + graphe
    fraud_sum_result = df.groupby('date').size()
    st.bar_chart(fraud_sum_result)

    return df

#Fonction pour filtrer les données avec un multislider et tracer un graphe
def filter_and_display(col,sidebar_title,mutiselect_label):
    
    st.sidebar.header(sidebar_title)

    #INPUT
    # Crée une liste des valeurs de la colonne sans doublons
    unique = sorted(filtered_data[col].unique())


    # Multiselect créé avec toutes les valeurs activées par défaut
    interest = st.sidebar.multiselect(mutiselect_label, default=None, options = unique)
    st.subheader(f"Nombre de paiements frauduleux en fonction de la variable {col}")

    # OUTPUT
    if not interest:
        # fraud_sum_result = filtered_data.groupby([col, 'is_fraud']).size().unstack(fill_value=0)
        fraud_sum_result = filtered_data.groupby(col).size()

        st.bar_chart(fraud_sum_result)
        df = filtered_data

    else:
        # Filtrer les données en fonction de la selection du multiselect
        df = filtered_data[filtered_data[col].isin(interest)]

        # Calcul du total de paiements frauduleux
        total_fraud_sum = len(df)
        st.sidebar.caption(f"{total_fraud_sum} résultats")

        # Somme de fraudes + graphe
        # fraud_sum_result = df.groupby([col, 'is_fraud']).size().unstack(fill_value=0)
        fraud_sum_result = df.groupby(col).size()
        st.bar_chart(fraud_sum_result)   

    return df
 

############################################ PROJET ############################################
st.sidebar.title("Outil de recherche avancée des fraudes")

# Charger les données dans Streamlit
uploaded_file = st.sidebar.file_uploader("Téléchargez un fichier CSV", type=["csv"])

if uploaded_file:
    # Lire les données
    filtered_data = pd.read_csv(uploaded_file)
    # st.write(filtered_data.head())

    #Données descriptives
    fraud_amt_sum = filtered_data['amt'].sum()
    st.sidebar.write(f"La somme des montants frauduleux est de {round(fraud_amt_sum, 2)} $")
    total_fraud_sum = filtered_data['is_fraud'].sum()
    st.sidebar.write(f"Le nombre total de paiements frauduleux est de {total_fraud_sum}.")


    
    col = 'date'
    st.subheader(f"Nombre de paiements frauduleux en fonction de la variable {col}")
    filtered_data = filter_by_date_with_slider(filtered_data) 

    filtered_data = filter_and_display("state","Selectionner les états :","Choisissez le(s) état(s) à étudier")

    filtered_data = filter_and_display("category","Selectionner la catégorie :","Choisissez le(s) categorie(s) à étudier")

    filtered_data = filter_and_display("group_age","Selectionner la tranche d'âge:","Choisissez le(s) groupe(s) d'âge à étudier")

    filtered_data = filter_and_display("gender","Selectionner le genre :","Choisissez le(s) genre(s) d'âge à étudier")

    filtered_data = filter_and_display("day","Nombre de fraudes en fonction du jour","Choisissez le(s) jour(s) à étudier")

    filtered_data = filter_and_display("hour","Nombre de fraudes en fonction de l'heure","Choisissez l'(es) heure(s) à étudier")


    st.subheader(f"Tableau récapitulatifs des transactions")
    filtered_data['cc_num'] = filtered_data['cc_num'].astype(str)
    simplified_data = filtered_data[['fulldate','fullname','state','city','category','amt','daily_amount$','daily_number_fraud','cc_num','iin']].sort_values(by= ['fullname','fulldate'])
    st.dataframe(simplified_data)












    