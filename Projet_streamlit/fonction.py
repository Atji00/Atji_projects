import pandas as pd
import numpy as np
import streamlit as st
# Fonction pour créer des tableaux statistiques croisés dynamiques
def cross_stat(data:pd.DataFrame = None, catlist:list = None, var:'str'=None):

    """--Docstring--
    fonction pour réaliser des statistiques
    sur des variables croisées.
    
    Args:
         data: le dataframe
         catlist: (list) variables catégorielles
         var: pandas.Series variable continue
    
    return a dataframe
    """
    # initialisation d'un tableau vide
    table = pd.DataFrame()
    
    # Création de dictionnaire pour les indexes
    index_mapping = {value: cat for cat in catlist for value in data[cat].unique()}

    for cat in catlist:
            
        X = data.groupby(by=cat)[var].agg(['min',
                                        'max',
                                        'mean',
                                        'std',
                                        lambda x: x.quantile(0.25),
                                        'median',
                                        lambda x: x.quantile(0.75),
                                        'count'])
        table = pd.concat([table,X], axis = 0, ignore_index = False)
    
    table.reset_index(names = 'valeurs', inplace = True)
    table['Categories'] = table['valeurs'].map(index_mapping)
    table.set_index(['Categories','valeurs'], inplace = True)
    table.columns = ['min','max','mean','st_deviation', 'quartile1',
                        'median','quartile3','category_size']
    return table.round(2)
