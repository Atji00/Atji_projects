import pandas as pd


def casting(variable: pd.Series, boolean_threshold: float = 0.95, numeric_threshold: float = 0.99, 
            datetime_threshold: float = 0.95, cardinality_relative: float = 0.05, 
            cardinality_absolute: int = 12) -> pd.Series:

    """---Cast a variable into an appropriate data type----

    The function attempts to convert the most appropriate data type
    for a variable. Object columns are tested successively for boolean,
    numeric, datetime, categorical, and string types.

    Numeric and datetime conversions are performed only when the
    proportion of successfully converted non-missing values reaches
    the corresponding threshold.

    Parameters
    ----------
    variable : pandas.Series Variable from raw data to be cast.

    boolean_threshold : float, default=0.95
        Minimum proportion of non-missing values that must be
        successfully converted to a boolean type for the conversion
        to be applied. Values that cannot be mapped are coerced
        to missing values during the conversion.

    numeric_threshold : float, default=0.99
            Minimum proportion of non-missing values that must be
            successfully converted to a numeric type for the conversion
            to be applied. Values that cannot be converted are coerced
            to missing values during the conversion.

    datetime_threshold : float, default=0.95
        Minimum proportion of non-missing values that must be
        successfully converted to datetime for the conversion
        to be applied. Values that cannot be converted are coerced
        to missing values during the conversion.

    cardinality_relative : float, default=0.05
        Maximum ratio of unique non-missing values to total
        non-missing values for a variable to be considered
        categorical.

    cardinality_absolute : int, default=12
        Maximum number of unique non-missing values for a variable
        to be considered categorical, regardless of its relative
        cardinality.

    Returns
    -------
    pandas.Series
        
    """

    variable_name = variable.name

    # ------------------------------------------------------------------
    # Empty variable / all values missing
    # ------------------------------------------------------------------

    if variable.dropna().empty:

        return variable

    # ------------------------------------------------------------------
    # Columns already correctly casted
    # ------------------------------------------------------------------

    if pd.api.types.is_integer_dtype(variable):
       
        return variable.astype("Int64")

    if pd.api.types.is_float_dtype(variable):
       
        return variable.astype("float64")

    if (
        pd.api.types.is_bool_dtype(variable)
        or pd.api.types.is_datetime64_any_dtype(variable)
        or isinstance(variable.dtype, pd.CategoricalDtype)
    ):
       
        return variable

    # ------------------------------------------------------------------
    # Object columns
    # ------------------------------------------------------------------

    if variable.dtype == "object":

        non_null = variable.dropna()
        non_missing = variable.notna()

        # ==============================================================
        # Boolean
        # ==============================================================

        mapping = {
                    "true": True,
                    "vrai": True,
                    "oui": True,
                    "yes": True,
                    "false": False,
                    "faux": False,
                    "non": False,
                    "no": False,
                }

        boolean = (variable.astype("string")
                              .str.strip()
                              .str.lower()
                              .map(mapping)
                              .astype("boolean")
                    )

        boolean_rate = boolean[non_missing].notna().mean()

        if boolean_rate >= boolean_threshold:

            print(
                    f"CASTED VARIABLE: {variable_name}\n"
                    f"-------------------\n"
                    f"Number of observations: {len(variable)}\n"
                    f"Number of non-missing values: {non_missing.sum()}\n"
                    f"Detected type: Boolean\n"
                    f"Conversion rate: {boolean_rate:.2%}\n"
                    f"Result dtype: {boolean.dtype}\n"
                )

            return boolean

        # ==============================================================
        # Numeric
        # ==============================================================

        numeric = pd.to_numeric(variable, errors="coerce")

        numeric_rate = numeric[non_missing].notna().mean()

        if numeric_rate >= numeric_threshold:

            if (numeric.dropna() % 1 == 0).all():

                result = numeric.astype("Int64")
                
            else:

                result = numeric.astype("float64")
                
            print(
                    f"CASTED VARIABLE: {variable_name}\n"
                    f"-------------------\n"
                    f"Number of observations: {len(variable)}\n"
                    f"Number of non-missing values: {non_missing.sum()}\n"
                    f"Conversion rate: {numeric_rate:.2%}\n"
                    f"Result dtype: {result.dtype}\n"
                 )

            return result

        # ==============================================================
        # Datetime
        # ==============================================================

        dates = pd.to_datetime(variable, errors="coerce")

        date_rate = dates[non_missing].notna().mean()

        if date_rate >= datetime_threshold:

            print(
                    f"CASTED VARIABLE: {variable_name}\n"
                    f"-------------------\n"
                    f"Number of observations: {len(variable)}\n"
                    f"Number of non-missing values: {non_missing.sum()}\n"
                    f"Datetime conversion rate: {date_rate:.2%}\n"
                    f"Result dtype: {dates.dtype}\n"
                )

            return dates

        # ==============================================================
        # Category / String
        # ==============================================================

        size = non_null.size
        unique_count = non_null.nunique()
        unique_ratio = unique_count / size

        if (
            unique_count <= cardinality_absolute
            or unique_ratio < cardinality_relative
        ):

            result = variable.astype("category")
            
        else:

            result = variable.astype("string")
            
        print(
                f"CASTING VARIABLE: {variable_name}\n"
                f"-------------------\n"
                f"Number of observations: {len(variable)}\n"
                f"Number of non-missing values: {non_missing.sum()}\n"
                f"Number of unique values: {unique_count}\n"
                f"Unique ratio: {unique_ratio:.2%}\n"
                f"Result dtype: {result.dtype}\n"
             )

        return result

    # ------------------------------------------------------------------
    # Other dtypes
    # ------------------------------------------------------------------

    print(
            f"NOT CASTED VARIABLE: {variable_name}\n"
            f"-------------------\n"
            f"dtype: {variable.dtype}\n"
            f"Result: unchanged\n"
        )

    return variable