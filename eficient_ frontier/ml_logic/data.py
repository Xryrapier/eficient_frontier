import pandas as pd
from pathlib import Path
from ml_logic.five_best_stock import *

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

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by
    - assigning  to each column Correct Data type
    - irrelevant transactions
    """


    print("✅ data cleaned")

    return df
