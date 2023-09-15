import pandas as pd
from pathlib import Path
from eficient_frontier.ml_logic.RTfuncs import *



def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by
    - assigning  to each column Correct Data type
    - irrelevant transactions
    """
    # Selecting some columns of interest
    selected_features = ['HHSEX',
                        'AGE',
                        'EDCL',
                        'MARRIED',
                        'KIDS',
                        'FAMSTRUCT',
                        'OCCAT1',
                        'INCOME',
                        'WSAVED',
                        'YESFINRISK',
                        'NETWORTH',
                        'LIQ',
                        'CDS',
                        'SAVBND',
                        'CASHLI',
                        'NMMF',
                        'STOCKS',
                        'BOND']

    # Overwriting the "data" variable to keep only the columns of interest
    data = data[selected_features].copy()

    #Adding a new column (target)
    data = calculate_risk_tolerance(data)

    # Dropping columns we don't need
    data.drop(columns = ["RiskFree"], inplace = True)
    data.drop(columns = ["Risky"], inplace = True)
    data.drop(columns = ["LIQ"], inplace = True)
    data.drop(columns = ["CDS"], inplace = True)
    data.drop(columns = ["SAVBND"], inplace = True)
    data.drop(columns = ["CASHLI"], inplace = True)
    data.drop(columns = ["NMMF"], inplace = True)
    data.drop(columns = ["STOCKS"], inplace = True)
    data.drop(columns = ["BOND"], inplace = True)

    # Dropping diplicates
    data = data.drop_duplicates()

    # Dropping missing data
    data = data.dropna(subset=['RT'])

    # Getting rid of outliers (very high income)
    q1_income = data['INCOME'].describe()['25%']
    q3_income = data['INCOME'].describe()['75%']

    iqr_income = q3_income - q1_income
    lower_bound = np.max([q1_income - 1.5*iqr_income,0])
    upper_bound = q3_income + 1.5*iqr_income

    normal_people = data['INCOME'] < upper_bound
    data_normal = data[normal_people]

    print("✅ data cleaned")

    return data_normal
