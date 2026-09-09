import pandas as pd


def information(data: pd.DataFrame, metadata:dict) -> pd.DataFrame:

    """_____Docstring_____\n
    Generate a summary of the dataset's structure and missing values.
 
    This function aims to Generate a summary of the dataset's structure: types, uniques values,
    Not Available and missing values.

    Parameters
    ----------
    data : pd.DataFrame
        data from raw folder
    
    metadata : dict
        which contains the same variables as data.csv and their definitions.   

    Returns
    -------
    pd.DataFrame
        A summary DataFrame containing each variable's name, data type,
        number of unique values, number and percentage of missing values,
        and its description.
    """

   
    data_variables = set(data.columns)

    metadata_variables = set(metadata)

    # data variables missing in metadata
    missing_in_metadata = data_variables - metadata_variables

    # metada variables missing in data
    missing_in_data = metadata_variables - data_variables
    

    if missing_in_data or missing_in_metadata:

        raise ValueError(f""" Variables not corresponding !\nMissing in Metadata: {missing_in_metadata}\n
                                Missing in Data: {missing_in_data}""")
    

    info_table = pd.DataFrame({
            'Variables': data.columns,
            'Type': data.dtypes,
            'Unique_values': data.nunique(),
            'NA_counts': data.isna().sum(),
            'NA_percent%':data.isna().mean().round(4)*100,
            }).reset_index(drop=True)

    info_table['Description_des_variables'] = info_table['Variables'].map(metadata)
        
    print(f"Taille: {data.shape[0]} - Nb Variables: {data.shape[1]}")

    return info_table