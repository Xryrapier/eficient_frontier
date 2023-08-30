import pandas as pd
from pathlib import Path
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
    selected_features = ['YY1',
                         'HHSEX',
                         'AGE',
                         'AGECL',
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

    # Drop diplicates
    data = data.drop_duplicates()

    # Data Transformation
    st_scaler = StandardScaler()
    data['AGE'] = st_scaler.fit_transform(data[['AGE']])

    rb_scaler = RobustScaler()
    data['KIDS'] = rb_scaler.fit_transform(data[['KIDS']])
    data['INCOME'] = rb_scaler.fit_transform(data[['INCOME']])
    data['NETWORTH'] = rb_scaler.fit_transform(data[['NETWORTH']])

    # Perform one-hot encoding on categorical columns
    ohe = OneHotEncoder(sparse_output = False)
    ohe.fit(data[['HHSEX']])
    ohe.categories_
    ohe.get_feature_names_out()
    data[ohe.get_feature_names_out()] = ohe.transform(data[['HHSEX']])
    data.drop(columns = ["HHSEX"], inplace = True)

    ohe.fit(data[['AGECL']])
    data[ohe.get_feature_names_out()] = ohe.transform(data[['AGECL']])
    data.drop(columns = ["AGECL"], inplace = True)

    ohe.fit(data[['EDCL']])
    data[ohe.get_feature_names_out()] = ohe.transform(data[['EDCL']])
    data.drop(columns = ["EDCL"], inplace = True)

    ohe.fit(data[['MARRIED']])
    data[ohe.get_feature_names_out()] = ohe.transform(data[['MARRIED']])
    data.drop(columns = ["MARRIED"], inplace = True)

    ohe.fit(data[['FAMSTRUCT']])
    data[ohe.get_feature_names_out()] = ohe.transform(data[['FAMSTRUCT']])
    data.drop(columns = ["FAMSTRUCT"], inplace = True)

    ohe.fit(data[['OCCAT1']])
    data[ohe.get_feature_names_out()] = ohe.transform(data[['OCCAT1']])
    data.drop(columns = ["OCCAT1"], inplace = True)

    ohe.fit(data[['WSAVED']])
    data[ohe.get_feature_names_out()] = ohe.transform(data[['WSAVED']])
    data.drop(columns = ["WSAVED"], inplace = True)

    print("✅ data cleaned")

    return data.shape
