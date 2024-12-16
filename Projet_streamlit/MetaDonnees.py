#===========================================================
# Importation des librairies
#===========================================================
import pandas as pd
import numpy as np
import streamlit as st
import time
import random
random.seed(123)

#===========================================================
# Chargement des données
#===========================================================

#Chargement de métadonnées1 sur les variables d'origine
metadonnees1 = pd.read_csv('data/metadonnees1.csv')

#Chargement de métadonnées2 sur les variables calculées
metadonnees2 = pd.read_csv('data/metadonnees2.csv')

#===========================================================
# Construction de la page
#===========================================================

lien = "https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset"

description = """Jeux de données sur les transactions horaires de cartes 
                de crédits et fraudes aux USA de 2019 à 2020.
            """
st.markdown("""<h1 style="color: darkblue; text-decoration: underline;
            ">Credit Card Transactions Dataset</h1>""", unsafe_allow_html=True
            )
st.markdown("<br>", unsafe_allow_html=True) #saut de ligne
col1,col2,col3, col4 = st.columns([2,1,1,2])

col1.subheader("Description", divider = True)
col1.markdown(f"""<div style="text-align: justify; color: darkblue;
                font-size: 20px;">{description}</div>""", unsafe_allow_html=True              
                )

col2.subheader("Taille", divider = True)
col2.markdown(f"""<div style="text-align: center; color: darkblue;
                font-size: 30px;">1.296.675</div>""", unsafe_allow_html=True              
                )
col3.subheader("Variables", divider = True)
col3.markdown(f"""<div style="text-align: center; color: darkblue;
                font-size: 30px;">23</div>""", unsafe_allow_html=True              
                )
col4.subheader("Source données", divider = True)
col4.page_link(lien, label="Kaggle Dataset", icon="🌎")

st.markdown("<br>", unsafe_allow_html=True) #saut de ligne
st.subheader("Tableau de synthèse sur les variables d'origine:")
st.dataframe(data=metadonnees1, hide_index=True,
            use_container_width = True)
st.subheader("Tableau de synthèse sur les nouvelles variables calculées:")
st.dataframe(data=metadonnees2, hide_index=True,
            use_container_width = True)
