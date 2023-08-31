import pandas as pd
from pathlib import Path
from RTfuncs import *
from eficient_frontier.ml_logic.five_best_stock import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder

def get_best_5_stock(tickers, sp500_data):
    """
    input= indexes from sp500
    output= list of five best stock
    """
    ladj = []
    print('Reading data:')

    for t in tqdm(sp500_data):
        ladj.append(t['adjclose'])

    df = pd.concat(ladj,axis=1,keys=tickers)
    clustered_df , silhouette = get_clustered_groups(ndays=10,nk=4)
    ntop = 10
    new_tickers = []
    for ic in [1]:
        iclus = clustered_df[ic]
        li = list(iclus.sort_values('Sharpe Ratio', ascending=False).head(ntop).index)
    for l in li:
        new_tickers.append(l)

    corr = silhouette*1.2
    res_portfolio =get_optimal_portfolio(new_tickers,ndays=10, corr_ratio=corr, only_corr=False)

    minRiskReturn = res_portfolio[0][1]['Return']
    minRiskWeights = res_portfolio[0][1]['Weights']

    res_portfolio[1].columns
    maxSharpeReturn = res_portfolio[0][2]['Return']
    maxSharpeWeights = res_portfolio[0][2]['Weights']
    return get_portfolio_stock_components(minRiskWeights, res_portfolio,df, investment=1e5)

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
