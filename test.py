import numpy as np
import pandas as pd
from IPython.display import display

from scoring_model.utils.func_descriptive_stats import *

data = pd.DataFrame(
    {
        "type_contrat": pd.Categorical(
            ["CDI", "CDD", "Interim", "CDI", "Freelance", "CDD", "CDI", "Interim", "CDI", "CDD"]
        ),
        "defaut_paiement": [False, True, False, False, True, False, False, True, False, False],
        "region": pd.Categorical(["Bretagne", "Ile-de-France", "Bretagne", "Occitanie", "Ile-de-France",
                    "Occitanie", "Bretagne", "Ile-de-France", "Occitanie", "Bretagne"]), 
        "montant_credit": [8500.0, 15200.0, 4300.0, 22000.0, np.nan, 9800.0, 12500.0, 3100.0, 17600.0, 6400.0],
        "duree_pret_mois": [36, 60, 24, 84, 48, 36, 60, 12, 72, 24],
        "revenu_mensuel": [2100.0, 3400.0, 1800.0, 4200.0, 2900.0, np.nan, 3100.0, 1650.0, 3800.0, 2000.0],
        "taux_endettement": [0.28, 0.35, 0.22, 0.41, 0.30, 0.19, 0.33, 0.25, 0.38, 0.21],
    }
)




if __name__ == "__main__":

    crossed_statistics(data, ["type_contrat", "region"], ["taux_endettement", "revenu_mensuel", "montant_credit"])  
