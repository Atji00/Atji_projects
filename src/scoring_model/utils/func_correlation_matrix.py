import warnings
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def correlation_matrix(data: pd.DataFrame, numeric_vars: list[str], 
                       method: Literal['pearson', 'spearman', 'kendall'] = 'kendall',
                       rotation_x: int = 45, ha_x: Literal['left', 'center', 'right'] = 'right', 
                       rotation_y: int = 45, ha_y: Literal['left', 'center', 'right'] = 'right') -> plt.Figure:

    """______Docstring_______\n
    Function to build a croorelation matrix for numeric variables

    Parameters
    ----------
    data: pd.DataFrame -> data form data/raw or processed

    numeric_vars: list[str] -> list of numeric variables from 'data'

    method: Literal['pearson', 'spearman', 'kendall'], default = 'kendall'
            method for correlation calculation
    
    rotation_x: int, default = 45, x_axis rotation angle

    rotation_y: int, default = 45, y_axis rotation angle

    ha_x: Literal['left', 'center', 'right'], default = 'right', x_axis orientation

    ha_y: Literal['left', 'center', 'right'], default = 'right', y_axis orientation

    Returns
    -------
    Returns a matplotlib figure object of the correlation matrix

    
    """

    # Authorized methods for correlation calculation
    methods = {'pearson', 'spearman', 'kendall'}

    # Initialize empty list of valid variables 
    valid_variables = []

    if method.lower() not in methods:

        raise ValueError(f"Invalid method '{method}'. Choose from {methods}.")
    

    for variable in numeric_vars:

        if variable not in data.columns:

            warnings.warn(f"Variable '{variable}' not found in DataFrame columns !\nPlease check again :)")

            continue
  
        if (not pd.api.types.is_numeric_dtype(data[variable]) 
            or pd.api.types.is_bool_dtype(data[variable])):

            warnings.warn(f"Error on {variable}: {data[variable].dtype} not <numeric dtype>")

            continue

        valid_variables.append(variable)

            
    if valid_variables:

        plt.figure(figsize=(8, 6))
        sns.heatmap(data[valid_variables].corr(method = method.lower()),
                    cmap='Blues', annot=True)
        plt.title(f"Correlations Matrix: method = {method}\n")
        plt.xticks(rotation = rotation_x, ha = ha_x)
        plt.yticks(rotation = rotation_y, ha = ha_y)
        plt.show()

    else:

        raise ValueError(f"No valid numeric variables found in {numeric_vars} !")

    return plt.gcf()