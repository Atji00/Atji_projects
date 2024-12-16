#===========================================================
# Importation des librairies
#===========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import time
import random
random.seed(123)
#===========================================================
# Chargement des données
#===========================================================

#chargement des des données individuelles
individual = pd.read_csv('data/individual_data_a.csv')


#===========================================================
# Construction de la page
#===========================================================

# Informations sur les détenteurs
st.markdown("""<h2 style="color: darkblue; text-decoration: underline;
            ">Identification du Détenteur de Carte de Crédits</h1>""",
            unsafe_allow_html=True
            )
identifiants = individual['personid'].unique().tolist()

id = st.selectbox("Selectionner ou saisir identifiant",
                    identifiants,  index=0)

persondata = individual[individual['personid']==id]

nomcomplet = persondata['fullname'].unique()[0]
profession = persondata['job'].unique()[0]
naissance = persondata['dob'].unique()[0]
age = persondata['age'].unique()[0]
sexe = persondata['gender'].unique()[0]
ville = persondata['city'].unique()[0]
carte = persondata['cc_num'].unique()[0]
emetteur = persondata['iin_group'].unique()[0]
adresse = persondata['street'].unique()[0]
card_dict = {'American Express':'images/americancard.png',
            'Visa':'images/visacard.png', 
            'JCB':'images/jcbcard.png',
            'Discover Financial Services':'images/discovercard.png',
            'Mastercard Group':'images/mastercard.png',
            'GPN':'images/gpncard.png',
            'UATP':'images/uatpcard.png',
            'Laser':'images/lasercard.png',
            'InstaPayment':'images/instapaymentcard.png'}
carteimage = card_dict[emetteur]

col1, col2, col3= st.columns(3)
space ='&nbsp;'

col1.subheader(f"{space*6}{nomcomplet}")
col1.image('images/cartebank.png')

col2.subheader(f"Infos")
col2.write(f"Nom: {nomcomplet}")
col2.write(f"Date de Naissance: {naissance}")
col2.write(f"Age: {age}{space*14}Sexe: {'Masculin' if sexe == 'M' else 'Feminin'}",
            unsafe_allow_html=True)
col2.write(f"Ville d'habitation: {ville}")
col2.write(f"Adresse: {adresse}")

col3.markdown(f"<h3 style='font-size:20px;'>{emetteur}</h3>", unsafe_allow_html=True)
col3.image(carteimage)
col3.write(f"Carte: {carte}")

st.markdown("<br>", unsafe_allow_html=True)  # Saut de ligne

st.subheader("Récapitulatif des transactions du Detenteur", divider='blue')
persondata['date'] = pd.to_datetime(persondata["date"]).dt.date
date_min = persondata['date'].min()
date_max = persondata['date'].max()

col1, col2 = st.columns(2)
debut = col1.date_input("selectionner date debut", date_min,
                        min_value = date_min, max_value = date_max)
fin = col2.date_input("selectionner date fin", date_max,
                        min_value = date_min, max_value = date_max)
persondata_filtre = (persondata[(persondata['date'] >= debut) & 
                        (persondata['date'] <= fin)])
transnumb = persondata_filtre['daily_transactions'].sum()
transamount = persondata_filtre['daily_amount$'].sum()
fraudnum = persondata_filtre['daily_number_fraud'].sum()
fraudamount = persondata_filtre['daily_fraud_amount$'].sum()


st.markdown(f"""<h3 style="font-size:25px; color: dark; text-align: center;"
                  >Periode: {date_min} au {date_max}</h3>""",unsafe_allow_html=True)
col1,col2= st.columns(2)
col1.markdown(f"""<h3 style="font-size:20px; color: royalblue; text-align: center;"
                  >Nombre de transactions{space*2}{transnumb:,.0f}</h3>""",
                  unsafe_allow_html=True)
col2.markdown(f"""<h3 style="font-size:20px; color: royalblue; text-align: center;"
                  >Total des montants{space*2}{transamount:,.2f} USD</h3>""",
                  unsafe_allow_html=True)

col1, col2 = st.columns(2)
col1.markdown(f"""<h2 style="font-size:20px; text-align: center;color:royalblue;"
                  >Evolution des montants journaliers:</h2>""",
                  unsafe_allow_html=True)
col1.area_chart(persondata_filtre[['date','daily_amount$']],
                x="date", y="daily_amount$", x_label= 'Periode',
                y_label='Montant USD')

col2.markdown(f"""<h2 style="font-size:20px; text-align: center; color:royalblue;"
                  >Evolution des transactions journalières:</h2>""",
                  unsafe_allow_html=True)
col2.bar_chart(persondata_filtre[['date','daily_transactions']],
                y="daily_transactions", x="date", x_label='Periode',
                y_label='Volume des transactions')

col1, col2 = st.columns(2)
col1.markdown(f"""<h3 style="font-size:20px; color: red; text-align: center;"
                  >Nombre de Fraudes{space*2}{fraudnum:,.0f}</h3>""",
                  unsafe_allow_html=True)
col2.markdown(f"""<h3 style="font-size:20px; color: red; text-align: center;"
                  >Total des montants{space*2}{fraudamount:,.2f} USD</h3>""",
                  unsafe_allow_html=True)

col1, col2 = st.columns(2)
col1.markdown(f"""<h2 style="font-size:20px; text-align: center; color: red;"
                  >Evolution des montants de fraudes journalières:</h2>""",
                  unsafe_allow_html=True)
col1.area_chart(persondata_filtre[['date','daily_fraud_amount$']],
                x="date", y="daily_fraud_amount$", x_label= 'Periode',
                y_label='Montant USD', color= "#ff0000")

col2.markdown(f"""<h2 style="font-size:20px; text-align: center; color: red;"
                  >Evolution des fraudes journalières:</h2>""",
                  unsafe_allow_html=True)

col2.bar_chart(persondata_filtre[['date','daily_number_fraud']],
                y="daily_number_fraud", x="date", x_label='Periode',
                y_label='Nombre de fraudes', color= "#ff0000")