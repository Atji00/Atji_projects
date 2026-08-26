import warnings
from collections.abc import Callable
from typing import Any

import pandas as pd
from IPython.display import display


def check_variables(data:pd.DataFrame, catlist: list[str], statvars: list[str]) -> tuple[list[str], list[str]]:

    """____Docstring___
    Intermediate function to check if variables present in each list
    are dataframe columns and its dtype.

    Parameters
    ----------
    data: pd.DataFrame -> data form data/raw or processed

    catlist: list[str] -> list of categorical variables from 'data'

    statvars: list[str] -> list of continous variables from 'data'

    Returns
    -------
    Returns a tuple object of valid list
    
    """

    cat_bool_var: Callable[[Any], bool] = lambda var: ( 
                                                    isinstance(var, pd.CategoricalDtype) 
                                                    or pd.api.types.is_bool_dtype(var)
                                                    )
    
    numeric_var: Callable[[Any], bool] = lambda var: (
                                                    pd.api.types.is_numeric_dtype(var) 
                                                    and not pd.api.types.is_bool_dtype(var)
                                                )

    def _select_valid_vars(varlist: list[str], dtype_check: Callable[[Any], bool], expected_types: str) -> list[str]:

                # Initialize empty list
                valid = []

                for var in varlist:

                    if var not in data.columns:

                        warnings.warn(f"Error on {var}: not in data columns")

                        continue

                    if not dtype_check(data[var].dtype):

                        warnings.warn(f"Error on {var}: {data[var].dtype} not in {expected_types}")

                        continue

                    valid.append(var)

                return valid

    catlist = _select_valid_vars(catlist, cat_bool_var, "(<category> or <boolean>)")
    statvars = _select_valid_vars(statvars, numeric_var, "(<integer> or <float>)")

    return catlist, statvars


def category_distribution(data: pd.DataFrame, list_variables: list[str]) -> pd.DataFrame:

    """_____Docstring_____
    Function makes a distribution table (frequency % and counts) of a categorical column.
 
    Parameters
    ----------
    data : pd.DataFrame data from raw/processed data
    
    list_variables : list[string] list of categorical variables from 'data'   

    Returns
    -------
    pd.DataFrame: A summary DataFrame of statisitics  
    """

    # Initialize empty list
    
    table = [] 

    # Check for empty <list_variables>
    if not list_variables:

        raise ValueError("list of variables empty !")


    for variable in list_variables:


        try:

            if variable in data.columns and (isinstance(data[variable].dtype, pd.CategoricalDtype)
                                             or pd.api.types.is_bool_dtype(data[variable].dtype)):

                counts = data[variable].value_counts()

                percentages = data[variable].value_counts(normalize=True) * 100

                distribution_table = pd.DataFrame({
                                                    "Size of observations": counts,
                                                    "Percentage (%)": percentages.round(2)
                                                })

                total_counts = counts.sum()
                total_percentage = percentages.sum()

                distribution_table.loc['Total'] = [total_counts, total_percentage]

                distribution_table.insert(0, 'Variable', variable)

                print(f"Summary Table for: {variable}")

                table.append(distribution_table)

            else:

                if variable not in data.columns:

                    raise KeyError(f"{variable} not found in data")

                else:

                    raise TypeError(f"{variable} is not a <dtype.category or dtype.bool>")

        except (KeyError, TypeError) as e:

            print(f"Error on variable <'{variable}'> : {e}")

    return pd.concat(table)



def crossed_statistics(data: pd.DataFrame, catlist: list[str], statvars: list[str]) -> dict:

    """____Docstring____\n
    Compute cross statistics between categorical variables and
    continuous variables.

    Args:
        data: (pd.DataFrame) from data raw/processed
        catlist: (list) names of categorical variables
        statvars: (list) names of continuous variables

    Returns:
        dictionary object: descriptive statistics for each continuous
        variables, grouped by each categorical variables
        example -> {var = 'Revenue': pd.DataFrame = stats_table for Revenue,
                    var = 'Spendings': pd.DataFrame = stats_table for Spendings}
    """

    catlist, statvars = check_variables(data, catlist, statvars)

    # Check for empty list
    if not catlist or not statvars:
        raise ValueError("One or both lists empty !")

    # Initialize empty dictionary

    Summary_tables = {}

    # Calculate summary statistics table for each continous variable in statvars
    for var in statvars:

        stats_table = []

        for cat in catlist:

            stats = data.groupby(by=cat, observed = True)[var].agg(
                min='min',
                max='max',
                mean='mean',
                st_deviation='std',
                quartile1=lambda x: x.quantile(0.25),
                median='median',
                quartile3=lambda x: x.quantile(0.75),
                category_size='count'
            )

            stats['Categories'] = cat
            stats['modalities'] = stats.index

            stats_table.append(stats)

        final_table = pd.concat(stats_table, ignore_index=True)

        # Organisation des colonnes
        final_table.set_index(['Categories', 'modalities'], inplace=True)

        # Record final_table for each var in summary dictionary

        Summary_tables[var] = final_table

        print(f"Tableau Statistiques croisées sur: {var}")
        display(final_table.round(2))
        print('\n\n')

    return Summary_tables