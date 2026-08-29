from typing import Literal

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer


def impute_missings(
                        data: pd.DataFrame,
                        numeric_strategy:  Literal["mean", "median", "knn"] = "median",
                        categorical_strategy: Literal["most_frequent", "constant", "knn"] = "most_frequent",
                        missing_indicator: bool = False,
                        knn_neighbors: int = 5,
                    ) -> pd.DataFrame:
    """____Docstring____\n
    Impute missings values from dataframe columns.

    Parameters
    ----------
    data : pd.DataFrame from raw/processed data

    numeric_strategy : Literal["mean", "median", "knn"], default="median"
        Authorized strategies for numeric columns
 
    categorical_strategy : Literal["most_frequent", "constant", "knn"], default="most_frequent"
        Authorized strategies for categorical columns

    add_indicator : bool, default=False
        Create missings indicator for each column containing missings values if set True
        Example : income -> income_missing_indicator

    knn_neighbors : int, default=5
        Number of neighbors when "knn strategy" choosed.

    Returns
    -------
    pd.DataFrame returned with new columns (missings indicator) and transformed columns

    """
    # Normalize None to NaN to avoid bug from SimpleImputer
    df = data.copy().replace({None:np.nan})

    # ---------------------------------------------------------
    # Parameters Validation
    # ---------------------------------------------------------

    valid_numeric_strategies = {"mean", "median", "knn"}

    valid_categorical_strategies = {"most_frequent", "constant", "knn"}

    if numeric_strategy not in valid_numeric_strategies:
        raise ValueError(
            f"{numeric_strategy} not in list {valid_numeric_strategies}"
        )

    if categorical_strategy not in valid_categorical_strategies:
        raise ValueError(
            f"{categorical_strategy} not in list {valid_categorical_strategies}"
        )

 
    if knn_neighbors < 1:
        raise ValueError(
            "Params: knn_neighbors must be greater ou equal to 1"
        )

    # ---------------------------------------------------------
    # Columns selection by types and missings
    # ---------------------------------------------------------

    numeric_columns = (
                        df.select_dtypes(include="number")
                          .columns[ df.select_dtypes(include="number")
                                      .isna()
                                      .any() ]
                          .tolist()
                       )

    categorical_columns = (
                            df.select_dtypes(exclude=["number", "datetime"])
                              .columns[df.select_dtypes(exclude = ["number", "datetime"])
                                         .isna()
                                         .any()]
                              .tolist()
                            )

    if missing_indicator:
        missing_columns = df.columns[df.isna().any()].tolist()

        for column in missing_columns:
            indicator_name = f"{column}_missing_indicator"

            df[indicator_name] = df[column].isna().astype("int64")

    # ---------------------------------------------------------
    # Numeric columns Imputing
    # ---------------------------------------------------------

    if numeric_columns:

        if numeric_strategy in {"median", "mean"}:

            imputer = SimpleImputer(strategy = numeric_strategy)

            df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

        else:

            imputer = KNNImputer(n_neighbors=knn_neighbors, weights="distance") 
                             

            df[numeric_columns] = imputer.fit_transform(df[numeric_columns])


    # ---------------------------------------------------------
    # Categorical/Boolean/objects columns Imputing
    # ---------------------------------------------------------
    
    if categorical_columns:

        if categorical_strategy == "most_frequent":

            imputer = SimpleImputer(strategy = categorical_strategy)

            df[categorical_columns] = imputer.fit_transform(df[categorical_columns])

        elif categorical_strategy == "constant":

            imputer = SimpleImputer(strategy = categorical_strategy, fill_value = "missing")

            df[categorical_columns] = imputer.fit_transform(df[categorical_columns])


        else:

            imputer = KNNImputer(n_neighbors = knn_neighbors, weights="distance")
                                 
            for col in categorical_columns:

                uniques = df[col].dropna().unique().tolist()

                mapping = {value:float(index+1) for index, value in enumerate(uniques)}
                
                reverse_mapping = {index:value for value, index in mapping.items()}

                df[[col]] = imputer.fit_transform(df[[col]].map(lambda x: mapping.get(x, x)))

                df[[col]] = (
                                df[[col]].round()
                                         .clip(
                                                lower=min(reverse_mapping),
                                                upper=max(reverse_mapping),
                                              )
                                         .astype("int64")
                                         .map(lambda x: reverse_mapping.get(x, x))
                            )               

    return df

