#===========================================================
# Importation des librairies
#===========================================================
import pandas as pd
import numpy as np
import streamlit as st
import time
import random
random.seed(123)

#==========================================================
# Construction des elements de la page
#==========================================================



st.markdown("""<h2 style="text-align: left; color: darkblue;
            text-decoration: underline;">About Us</h2>""",
            unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True) #saut de ligne

col1, col2 = st.columns([3, 2])
col2.image("images/logo.png")

text1 = """Créé le 1er Octobre 2021, SDA est spécialisé dans la fourniture 
            de services Cloud, d'IA générative, de Machine Learning et 
            de construction d'infrastructures de données. Nous possédons 
            des experts et spécialistes métiers dans: la Banque-finance, 
            l'Agriculture, la Santé et l'économie. Notre modèle business
            innovant s'appuie principalement sur la capitalisation de l'expérience
            client, ce qui nous permet d'améliorer continuellement nos services et 
            produits fournis."""

col1.markdown(f"""<div style="text-align: justify; color: darkblue; font-size: 20px;"
                >{text1}</div>""", unsafe_allow_html=True
                )

st.markdown("<br>", unsafe_allow_html=True)  # Saut de ligne 

st.markdown("""<h2 style="text-align: center; 
                color: darkblue;
                ">Expertises Métiers</h2>""", 
                unsafe_allow_html=True
                )

col1, col2, col3, col4 = st.columns(4)
col1.image("images/bank.jpg")
col1.markdown(f"""<div style="text-align: justify;color: darkblue;
            ">BANQUE</div>""",unsafe_allow_html=True)
col2.image("images/economie.jpg")
col2.markdown(f"""<div style="text-align: justify;color: blue;
            ">ECONOMIE</div>""",unsafe_allow_html=True)
col3.image("images/sante.jpg")
col3.markdown(f"""<div style="text-align: justify;color: steelblue;
            ">SANTE</div>""",unsafe_allow_html=True)
col4.image("images/agriculture.jpg")
col4.markdown(f"""<div style="text-align: justify;color: green;
            ">AGRICULTURE</div>""",unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)  # Saut de ligne 

st.markdown("""<h2 style="text-align: center; color: darkblue;
                ">Présentation de l'équipe</h2>""", 
                unsafe_allow_html=True
                )

col1, col2, col3, col4 = st.columns(4)

col1.subheader("Co-founder")
col1.image("images/fanta.jpg")
text1 = """Titulaire d'un double diplôme en MBA Audit, Contôle
            de Gestion & Data Analytics, Mme Sanogo Fanta est spécialiste
             en risques de marchés, de crédits et financements structurés."""
col1.markdown(f"""<div style="text-align: justify;">{text1}</div>""",
                unsafe_allow_html=True
                )

col2.subheader("Co-founder")
col2.image("images/ilias.jpg")
text2 = """Fort d'une expérience de 7 ans chez Barclays, en qualité 
           de Data Engineer Senior, M. Ilias Benlarbi est responsable 
           de l'architecture des données clients de SDA."""
col2.markdown(f"""<div style="text-align: justify;">{text2}</div>""",
                unsafe_allow_html=True
                )

col3.subheader("President")
col3.image("images/robin.jpg")
text3 = """Titulaire d'un PhD. en Finance Internationale & Expert
            en Risques pays et bancaires, M. Robin Wabo préside
            SDA Group."""
col3.markdown(f"""<div style="text-align: justify;">{text3}</div>""",
                unsafe_allow_html=True
                )

col4.subheader("Director")
col4.image("images/atji.jpg") 
text4 = """Statisticien Senior & Data Analyst, M. Atji Cheick
           est spécialiste en Analyse Quantitative de Risques."""
col4.markdown(f"""<div style="text-align: justify;">{text4}</div>""",
                unsafe_allow_html=True
                )
