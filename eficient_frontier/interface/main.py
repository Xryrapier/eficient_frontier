import numpy as np
import pandas as pd
from pathlib import Path
from dateutil.parser import parse
from eficient_frontier.ml_logic.data import *
from eficient_frontier.ml_logic.preprocessor import *
from tqdm import tqdm
from eficient_frontier.ml_logic.finfuncs import *
import sys
import traceback
import ipdb
def get_actions_opt_portfolio(ndays, invest, sigma = None):
    """
    Generate an optimal portfolio of stocks based on given criteria.

    This function calculates an optimal portfolio of stocks using a combination of clustering,
    returns, and other financial metrics. It returns a DataFrame containing the selected stocks,
    the number of actions to invest in each stock, and their corresponding weights.

    Parameters:
    - ndays (int): Number of days to consider for analysis.
    - invest (float): The total amount available for investment.
    - sigma (float, optional): The desired standard deviation for the portfolio's risk.

    Returns:
    - opt_portfolio (pd.DataFrame): DataFrame with columns 'Ticker', 'Number of actions', and 'Weight'
      representing the selected stocks, the number of actions to invest in each stock, and their weights.

    If `sigma` is provided, an additional DataFrame `sigma_pd` is returned for a portfolio
    that aims to achieve the specified standard deviation (Risk Tolerance).

    Example:
    - opt_portfolio, sigma_pd = get_actions_opt_portfolio(ndays=30, invest=10000, sigma=0.2)
    """

    tickers, sp500_data = get_sp500_data()
    ladj = []
    print('Reading data:')

    for t in tqdm(sp500_data):
        ladj.append(t['adjclose'])

    df = pd.concat(ladj,axis=1,keys=tickers)

    nk = 4
    clustered_df = get_clustered_groups(ndays=ndays,nk=nk, plot=False)

    tot_top = 20
    new_tickers = []
    n_ind = []
    for i in range(nk):
        if clustered_df[i]['Returns'].mean() > 0:
            n_ind.append(i)

    ntop = tot_top // len(n_ind)

    for ic in n_ind:
        iclus = clustered_df[ic]
        li = list(iclus.sort_values('Sharpe Ratio', ascending=False).head(ntop).index)
        for l in li:
            new_tickers.append(l)

    res_portfolio= get_optimal_portfolio(new_tickers,ndays=ndays)


    minRiskReturn = res_portfolio[0][1]['Return']
    minRiskWeights = res_portfolio[0][1]['Weights']

    res_portfolio[1].columns
    maxSharpeReturn = res_portfolio[0][2]['Return']
    maxSharpeWeights = res_portfolio[0][2]['Weights']

    sel_tickers =  list(res_portfolio[1].columns)
    n_actions = get_portfolio_stock_components(minRiskWeights,sel_tickers, df, investment = invest)
    opt_portfolio = pd.DataFrame(np.vstack((sel_tickers,n_actions,np.round(minRiskWeights,2) )).T, columns = ['Ticker','Number of actions', 'Weight'] )

    if sigma :
        dif = abs(res_portfolio[0][0]['Standard Deviation'] - sigma)
        dif.name = 'Dif'
        new_pd = pd.concat([res_portfolio[0][0], dif], axis = 1)
        sigma_pd = new_pd.sort_values(by='Dif').head(1)
        n_actions2 = get_portfolio_stock_components(np.array(sigma_pd['Weights'].to_list())[0],sel_tickers, df, investment = invest)
        sigma_pd = pd.DataFrame(np.vstack((sel_tickers,n_actions2,np.round(np.array(sigma_pd['Weights'].to_list()),2) )).T, columns = ['Ticker','Number of actions', 'Weight'] )

        return opt_portfolio, sigma_pd
    else:
        return opt_portfolio

    print("\n ⭐️ get_actions_opt_portfolio")

def preproces_train_evaluate_save_model():
#     """
#     - Clean and preprocess data
#     - Train a model on it
#     - Save the model
#     - Compute & save a validation performance metric
#     """
    print("\n ⭐️ Loading Data")
    path = os.path.expanduser('~')+'/code/Xryrapier/eficient_frontier/raw_data/SCFP2019.csv'
    data = pd.read_csv(path)
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

    data = data[selected_features].copy()

    print("✅ Loading Data done")
    print("\n ⭐️ preprocessing and training and evaluating and saving model done")

    data=clean_data(data)
    X_train_preprocessed, X_test_preprocessed, y_train, y_test, preprocessor = preprocess(data)
    model= create_and_fit_model(X_train_preprocessed, y_train,\
        ExtraTreesRegressor, n_estimators=300, max_depth=30, min_samples_split=2, min_samples_leaf=1)
    mse, r2 = evaluate_regression_model(model, X_test_preprocessed, y_test)
    print("Mean Squared Error:", mse)
    print("R^2 Score:", r2)
    model_filename = 'models/finalized_model.sav'
    preprocessor_filename = 'models/finalized_preprocessor.sav'
    save_model_to_pickle(model, model_filename)
    save_model_to_pickle(preprocessor, preprocessor_filename)
    print("✅ preprocessing and training and evaluating and saving model done")


def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    print("\n ⭐️  loading model and prediction" )
    model_path_name = 'models/finalized_model.sav'

    model = load(open(model_path_name, 'rb'))
    preprocessor_path_name ='models/finalized_preprocessor.sav'
    preprocessor = load(open(preprocessor_path_name, 'rb'))

    X_pred_preprocessed = preprocessor.transform(X_pred)
    y_pred = model.predict(X_pred_preprocessed)

    print(f"✅ loading model and prediction done")
    return y_pred


if __name__ == '__main__':
    try:
        update_sp500_data()
    except:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
