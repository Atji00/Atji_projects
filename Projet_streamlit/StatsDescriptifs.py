#===========================================================
# Importation des librairies
#===========================================================
import pandas as pd
import numpy as np
import streamlit as st
import time
import random
import fonction as fc
random.seed(123)

#===========================================================
# Chargement des données
#===========================================================

individual = pd.read_csv('data/individual_data_a.csv')

financial = pd.read_csv('data/financial_clean.csv')

#==========================================================
# Construction de la page
#==========================================================

st.markdown("""<h1 style="color: darkblue; text-decoration: underline;"
            >Tableau statistiques croisées</h1>""", unsafe_allow_html=True)

col1,col2 = st.columns(2)
data_dic = {'financial_data':financial,
            'individual_data':individual}

data_list = list(name for name in data_dic.keys())
data_name = col1.selectbox("Choisir un Dataframe dans la liste:",
                        data_list, index=0, #dataframe par défaut
                        placeholder="select data...")

data = data_dic[data_name]
data['date'] = pd.to_datetime(data["date"]).dt.date
date_min = data['date'].min()
date_max = data['date'].max()

# liste des variables
num_list = data.select_dtypes(include=['number']).columns.tolist()
cat_list = data.select_dtypes(include=['object', 'category']).columns.tolist()

# Variables catégorielles sélectionnées
catvars = col1.multiselect("Choisir une ou plusieurs variables catégorielles:",
                            cat_list, default = ['group_age'])

# variable sélectionnée dans num_list
var = col1.selectbox("Choisir une variable numerique:", num_list,
                    index=num_list.index('daily_amount$'),
                    placeholder="select data...")
# Filtres sur les dates
debut = col2.date_input("selectionner date debut", date_min,
                        min_value = date_min, max_value = date_max)
fin = col2.date_input("selectionner date fin", date_max,
                        min_value = date_min, max_value = date_max)
try:

    # Filtrer les données sur les plages de dates
    data_filtre = (data[(data['date'] >= debut) & 
                            (data['date'] <= fin)]
                    )
    table = fc.cross_stat(data_filtre, catvars,var) 
    st.write("Statistiques par catégories sur:",var)
    st.dataframe(table, use_container_width = True)

except Exception as e:
    st.markdown("*Entrer des valeurs dans les champs* 😊")
