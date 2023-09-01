import numpy as np
import pandas as pd
from pathlib import Path
from dateutil.parser import parse
from ml_logic.data import *
from ml_logic.preprocessor import *
from ml_logic.model import *
from tqdm import tqdm
from eficient_frontier.ml_logic.finfuncs import get_sp500_data, get_clustered_groups, get_optimal_portfolio, get_portfolio_stock_components

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

    #print ('Analysis resutls:')
    #print ('Tickers: ', list(res_portfolio[1].columns))
    #print ('Minimum risk return and weights:', minRiskReturn, minRiskWeights)
    #print ('Max Sharpe ratio return and weights:', maxSharpeReturn, maxSharpeWeights)
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

# def preprocess_and_train():
#     """
#     - Clean and preprocess data
#     - Train a Keras model on it
#     - Save the model
#     - Compute & save a validation performance metric
#     """

#     print("\n ⭐️ preprocess_and_train")


#     # Clean data using data.py
#     data=clean_data(data)

#     # Create (X_train, y_train, X_val, y_val) without data leaks


#     # Create (X_train_processed, X_val_processed) using `preprocessor.py`
#     X_train_processed= preprocess_features("X_train")
#     X_val_processed=preprocess_features("X_val")

#     # Train a model on the training set, using `model.py`
#     model = None
#     model = initialize_model(input_shape=X_train_processed.shape[1:])
#     model , history = train_model(model,X_train_processed,"y_train" , validation_data=(X_val_processed,"y_val"))

#     # Save trained model
#     pass #your code here

#     print("✅ preprocess_and_train() done")


# def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
#     print("\n ⭐️  pred" )

#     if X_pred is None:
#         X_pred = pd.DataFrame(dict(
#             # defualt data to predict
#         ))

#     model = load_model()
#     X_processed = preprocess_features(X_pred)
#     y_pred = model.predict(X_processed)

#     print(f"✅ pred() done")

#     return y_pred


if __name__ == '__main__':
    try:
        tickers, sp500_data = get_sp500_data()
        get_best_5_stock(tickers, sp500_data)
    except:
        import sys
        import traceback

        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
