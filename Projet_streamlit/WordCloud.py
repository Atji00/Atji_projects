import streamlit as st
import pickle
import plotly.express as px

def load_wordcloud():
    """Charge le Word Cloud sauvegardé depuis le fichier Pickle."""
    try:
        with open("wordcloud.pkl", "rb") as f:
            wordcloud = pickle.load(f)
        return wordcloud
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Erreur : {e}. Assurez-vous que le fichier 'wordcloud.pkl' existe.")

# Charger le Word Cloud
wordcloud = load_wordcloud()

# Convertir le Word Cloud en tableau d'images (numpy array)
wordcloud_image = wordcloud.to_array()

# Utiliser Plotly pour afficher l'image
fig = px.imshow(
    wordcloud_image,
    color_continuous_scale='cool',  # Palette de couleurs
)

# Suppression des marges et des bordures blanches
fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),  # Pas de marges
    coloraxis_showscale=False  # Pas d'échelle des couleurs
)

# Suppression des ticks des axes
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)

# Streamlit interface

st.write("Voici notre Word Cloud interactif généré à partir de vos données :")
st.plotly_chart(fig)



#####===========================================================

import streamlit as st
from wordcloud import WordCloud
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import io

# Fonction pour nettoyer le texte en supprimant le HTML
def clean_html(raw_html):
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text()

# Fonction pour générer un WordCloud personnalisé
def generate_wordcloud(word_weights, width, height, background_color, colormap, font_path):
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        colormap=colormap,
        max_words=100,  # Limiter à 100 mots pour un visuel clair
        contour_width=1,
        contour_color='steelblue',
        font_path=font_path
    ).generate_from_frequencies(word_weights)
    return wordcloud

# Interface utilisateur
st.title("Générateur de WordCloud Personnalisable")
st.write("Chargez un article pour générer un WordCloud personnalisé avec vos options.")

# Options de personnalisation
background_color = st.color_picker("Choisissez une couleur de fond :", "#ffffff")
width = st.slider("Largeur du WordCloud :", min_value=400, max_value=1200, value=800, step=50)
height = st.slider("Hauteur du WordCloud :", min_value=200, max_value=800, value=400, step=50)
colormaps = [
    "viridis", "plasma", "inferno", "magma", "cividis", "cool", "spring",
    "summer", "autumn", "winter"
]
colormap = st.selectbox("Sélectionnez une palette de couleurs :", colormaps)

fonts = ["sans-serif", "serif", "monospace", "Comic Sans MS", "Arial", "Verdana"]
font = st.selectbox("Choisissez une police :", fonts)
font_path = None  # Optionnel : spécifiez un chemin si une police externe est utilisée.

# Chargement du fichier texte
uploaded_file = st.file_uploader("Chargez votre fichier texte (format TXT)", type=["txt"])
if uploaded_file:
    # Lecture du fichier texte
    raw_text = uploaded_file.read().decode("utf-8")

    # Nettoyage du contenu HTML
    text = clean_html(raw_text)
    st.write("Aperçu du contenu chargé (après nettoyage) :")
    st.text_area("Contenu de l'article", text, height=200)

    # Prétraitement du texte
    words = text.split()
    tfidf_df = pd.DataFrame(words, columns=["Mot"])
    
    # Calcul des occurrences et normalisation
    tfidf_df = tfidf_df.groupby("Mot").size().reset_index(name="Occurrences")
    tfidf_df["Fréquence"] = tfidf_df["Occurrences"] / tfidf_df["Occurrences"].sum()

    # Filtrage des mots
    filtered_tfidf_df = tfidf_df[
        (tfidf_df['Fréquence'] > 0.005) &  # Fréquence minimale ajustée pour plus de mots
        (tfidf_df['Fréquence'] < 0.2) &  # Fréquence maximale ajustée pour éviter les mots trop fréquents
        (tfidf_df['Mot'].str.len().between(4, 20)) &  # Longueur des mots élargie
        (~tfidf_df['Mot'].str.contains(r'[^a-zA-Zéèàùç]', regex=True))  # Exclusion des caractères spéciaux
    ]
    
    # Vérification si des mots pertinents existent
    if filtered_tfidf_df.empty:
        st.warning("Aucun mot valide n'a été trouvé après le filtrage. Vérifiez votre fichier ou ajustez les critères.")
    else:
        word_weights = dict(zip(filtered_tfidf_df['Mot'], filtered_tfidf_df['Fréquence']))

        # Génération du WordCloud
        wordcloud = generate_wordcloud(word_weights, width, height, background_color, colormap, font_path)

        # Affichage du WordCloud
        st.write("Votre WordCloud :")
        fig, ax = plt.subplots(figsize=(width / 100, height / 100))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

        # Téléchargement du WordCloud
        buffer = io.BytesIO()
        wordcloud.to_image().save(buffer, format="PNG")
        buffer.seek(0)
        st.download_button(
            label="Télécharger le WordCloud",
            data=buffer,
            file_name="wordcloud.png",
            mime="image/png"
        )

