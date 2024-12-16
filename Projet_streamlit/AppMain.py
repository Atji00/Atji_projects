#===========================================================
# Importation des librairies
#===========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import cufflinks as cf
import streamlit as st
import time
import random
random.seed(123)

#==========================================================
# Configuration de la page streamlit 
#==========================================================
st.set_page_config(
    page_title="Credit Card Dashboards",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state = "expanded"
)

#==========================================================
# Execution des pages
#==========================================================

# Structure des pages de l'application web
pages = [
    st.Page("AboutUs.py", title="Home"),
    st.Page("MetaDonnees.py", title="MétaDonnées"),
    st.Page("StatsDescriptifs.py", title="Statistiques Descriptives"),
    st.Page("TableauBordIndividu.py", title="Suivi des KPIs Individus"),
    st.Page("GeoAnalysisCharts.py", title="GeoAnalyse"),
    st.Page("MacroAnalyseCharts.py", title="MacroAnalyse"),
    st.Page("WordCloud.py", title="WordCloud")
]

pg = st.navigation(pages)
pg.run()
