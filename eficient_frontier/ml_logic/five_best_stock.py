
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import cvxpy as cp
from sklearn import metrics
import warnings
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay
import random
import matplotlib as mpl
import seaborn as sns
from ml_logic.sp500_data import get_sp500_data


def get_clustered_groups (ndays=180, plot=True, nk = 4):
    # load tickers and sp500 data and get dataframe with close prices
    tickers, sp500_data = get_sp500_data()
    ladj = []
    print('Reading data:')

    for t in tqdm(sp500_data):
        ladj.append(t['adjclose'][-ndays:])

    df = pd.concat(ladj,axis=1,keys=tickers)

    ladj2=[]
    for t in tqdm(sp500_data):
        ladj2.append(t['volume'][:])

    df_vol = pd.concat(ladj2,axis=1,keys=tickers)
    volch = df_vol.mean()
    # compute returns and std dev
    returns = df.pct_change().mean() * 252 # number of working days in a year
    std = df.pct_change().std() * np.sqrt(252)
    ret_var = pd.concat([returns, std], axis = 1).fillna(method='backfill')
    ret_var.columns = ["Returns","Standard Deviation"]
    Vol = volch.fillna(method='backfill').values
    Vol = Vol/np.max(Vol)
    # Get the indices of the top 20 stocks by volume
    top_stock_indices = np.argsort(Vol)[::-1][:8]

    # Retrieve the ticker symbols associated with the top indices
    top_stock_tickers = [tickers[i] for i in top_stock_indices]
    # compute k-means with different number of clusters:
    X = ret_var.values
    print(type(volch), len(volch), Vol.shape,X.shape)
    sse = []
    for k in range(1,15):
        kmeans = KMeans(n_clusters = k)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)

    if plot:
        plt.plot(range(1,15), sse)
        plt.xlabel("Value of k")
        plt.ylabel("Distortion")
        plt.show()

    X = ret_var.values
    kmeans =KMeans(n_clusters = nk).fit(X)
    centroids = kmeans.cluster_centers_
    print(type(volch), len(volch), Vol.shape,X.shape)
    if plot:
        plt.figure(figsize=(10,10))
        scatter = plt.scatter(X[:,0],X[:,1], c = kmeans.labels_, cmap ="viridis", s=150*Vol)
        plt.xlabel("returns", fontsize=16)
        plt.ylabel("std dev",fontsize = 16)
        plt.scatter(centroids[:,0], centroids[:,1],color="red",marker="*")

        for i, ticker in enumerate(top_stock_tickers):
            plt.annotate(ticker, (X[top_stock_indices[i], 0], X[top_stock_indices[i], 1]),
                     textcoords="offset points", xytext=(0,10), ha='center',fontsize=11)
        # Crear una leyenda personalizada
        nk=nk+1
        print (nk)
        cbar = plt.colorbar(scatter, ticks=np.arange(1,nk,1))
        cbar.set_label('Groups')  # Etiqueta de la leyenda

        plt.show()

    y_predkmeans = pd.DataFrame(kmeans.predict(X))
    y_predkmeans = y_predkmeans.dropna()
    silhouette= metrics.silhouette_score(X,y_predkmeans)
    stocks = pd.DataFrame(ret_var.index)
    cluster_labels = pd.DataFrame(kmeans.labels_)
    stockClusters = pd.concat([stocks, cluster_labels],axis = 1)
    stockClusters.columns = ['Symbol','Cluster']
    x_df = pd.DataFrame(X, columns = ["Returns", "Volatility"])
    closerv = pd.concat([stockClusters,x_df],axis=1)
    closerv = closerv.set_index("Symbol")
    closerv['Sharpe Ratio'] = (closerv['Returns'] ) / closerv['Volatility']
    grouped = closerv.groupby('Cluster')

    # Dictionary with clustered dataframes
    cluster_dataframes = {}

    for cluster, group in grouped:
        cluster_dataframes[cluster] = group.drop('Cluster', axis=1)

    return cluster_dataframes , silhouette


def efficient_frontier_from_df(df, plot=True,npts = 80):
    returns = df.pct_change().mean() * (252)
    std = df.pct_change().std() * np.sqrt(252)

    # Calcular la matriz de covarianza
    cov_matrix = df.pct_change().cov() * (252)

    # Definir el rango de rendimientos objetivos
    target_returns = np.linspace(returns.min(), returns.max(), num=npts)

    # Calcular la frontera eficiente
    efficient_frontier = []
    for target_return in target_returns:
        weights = cp.Variable(len(df.columns))
        risk = cp.sqrt(cp.quad_form(weights, cov_matrix))
        constraints = [cp.sum(weights) == 1, cp.sum(weights @ returns) == target_return, weights >=0]
        try:
            problem = cp.Problem(cp.Minimize(risk), constraints)
            problem.solve(qcp=True)

            efficient_frontier.append((target_return, risk.value, weights.value))
        except:
            warnings.warn('unable to solve problem')

    # Convertir la lista en un DataFrame
    efficient_frontier_df = pd.DataFrame(efficient_frontier, columns=["Return", "Standard Deviation", "Weights"])

    # Encontrar el punto de menor desviación estándar
    min_std_idx = np.argmin(efficient_frontier_df["Standard Deviation"])
    min_std_point = efficient_frontier_df.iloc[min_std_idx]

    # Encontrar el punto de máximo Sharpe Ratio
    sharpe_ratios = efficient_frontier_df["Return"] / efficient_frontier_df["Standard Deviation"]
    max_sharpe_idx = np.argmax(sharpe_ratios)
    max_sharpe_point = efficient_frontier_df.iloc[max_sharpe_idx]

    # Calcular los pesos de Weighted Sharpe Ratio
    weighted_sharpe_ratios = np.vstack(np.array(sharpe_ratios) * np.abs(np.array(efficient_frontier_df["Weights"]))).sum(axis=1)
    max_weighted_sharpe_idx = np.argmax(weighted_sharpe_ratios)
    max_weighted_sharpe_point = efficient_frontier_df.iloc[max_weighted_sharpe_idx]


    #if plot:
    #    # Create a Seaborn style context
    #    sns.set(style="whitegrid")
#
    #    # Create the figure and axis
    #    plt.figure(figsize=(10, 6))
#
    #    # Plot the efficient frontier and key points using Seaborn functions
    #    sns.lineplot(data=efficient_frontier_df, x="Standard Deviation", y="Return", marker='o', label='Efficient Frontier')
    #    plt.scatter(min_std_point[1], min_std_point[0], marker='o', color='r', label='Min Std Deviation')
    #    sns.scatterplot(data=max_sharpe_point, x="Standard Deviation", y="Return", marker='o', color='g', label='Max Sharpe Ratio')
    #    sns.scatterplot(data=max_weighted_sharpe_point, x="Standard Deviation", y="Return", marker='o', color='b', label='Max Weighted Sharpe Ratio')
    #    plt.scatter(std, returns, marker='x', color='g', label='Actual Assets')
#
    #    # Set labels and title
    #    plt.xlabel('Standard Deviation')
    #    plt.ylabel('Expected Return')
    #    plt.title('Efficient Frontier and Key Points')
#
    #    # Add legend and show the plot
    #    plt.legend()
    #    plt.show()

    if plot:
        # Graficar la frontera eficiente y los puntos destacados
        plt.figure(figsize=(10, 6))
        plt.plot(efficient_frontier_df["Standard Deviation"], efficient_frontier_df["Return"],color='0.8', marker='o', label='Efficient Frontier')
        plt.plot(min_std_point[1], min_std_point[0], marker='h', color='r', label='Min Std Deviation')
        plt.plot(max_sharpe_point["Standard Deviation"], max_sharpe_point["Return"], marker='X', color='c', label='Max Sharpe Ratio')
        plt.plot(max_weighted_sharpe_point["Standard Deviation"], max_weighted_sharpe_point["Return"], marker='o', color='b', label='Max Weighted Sharpe Ratio')
        plt.scatter(std, returns, marker='x', color='g', label='Portfolio Stocks')
        plt.xlabel('Standard Deviation ', fontsize=16)
        plt.ylabel('Expected Return (annualized)', fontsize=16)
        plt.title('Efficient Frontier', fontsize = 16)
        plt.legend(fontsize=12)
        plt.show()

    return [efficient_frontier_df, min_std_point, max_sharpe_point, max_weighted_sharpe_point]

def get_optimal_portfolio(ticker_list, ndays=180, corr_ratio = 0.5, only_corr=False, ref_date = None):
    # get data
    tickers, sp500_data = get_sp500_data()
    ladj2 = []
    ltick = []
    tkr_list = ticker_list.copy()
    print('Preparing data...')
    #if not ref_date :
    #    datefin = pd.to_datetime(ref_date)
    #    dateini = datefin - BDay(ndays)
    #    for t in tqdm(sp500_data):
    #        if t['ticker'][0] in tkr_list:
    #            ltick.append(t['ticker'][0])
    #            ladj2.append(t['adjclose'][dateini:datefin])
   # else:
    for t in tqdm(sp500_data):
        if t['ticker'][0] in tkr_list:
            ltick.append(t['ticker'][0])
            ladj2.append(t['adjclose'][-ndays:])

    df = pd.concat(ladj2,axis=1,keys=ltick)

    # Correlation matrix
    correlation_matrix = df.corr(method="pearson")
    correlation_threshold = corr_ratio # Ajusta el umbral según tus preferencias
    combinations = list(itertools.combinations(ltick, 5))
    print(len(combinations))

    selected_dataframes = []
    corr_means = []
    print('Evaluating correlation of combinations')
    for  combo in tqdm(combinations):
        selected_columns = list(combo)
        combo_correlation = correlation_matrix.loc[selected_columns, selected_columns]
        corr_means.append(combo_correlation.values.mean())
        if (combo_correlation.values.mean() <= correlation_threshold):
            combo_df = df[selected_columns].copy()
            selected_dataframes.append(combo_df)

    print("Number of selected combinations:", len(selected_dataframes))
    print('Minimum correlation: ', np.min(np.array(corr_means)))
    if only_corr:
        return np.min(np.array(corr_means)), len(selected_dataframes)
    else:
        gather_results=[]
        print('Evaluating CLA for each combination')
        for itick in tqdm(selected_dataframes):
            res = efficient_frontier_from_df(itick, plot=False, npts = 40)
            gather_results.append([res[1]])
        returns = []
        for g in gather_results:
            returns.append(g[0]['Return'])
        idx = np.argmax(returns)
        res = efficient_frontier_from_df(selected_dataframes[idx], plot=True, npts = 80)
        return res, selected_dataframes[idx],

def get_portfolio_stock_components(minRiskWeights, res_portfolio,df, investment=1e5):
    sel_tickers = list(res_portfolio[1].columns)
    prices = []
    for tick in sel_tickers:
        prices.append(df[tick][-1])
    n_actions = []

    for i in range(5):
        amount = investment * minRiskWeights[i]
        nac = round(amount/prices[i])
        n_actions.append(nac)
    n_actions
    print ('The advise is to invest:')
    for i in range(5):
        print('%4i actions of %6s'%(n_actions[i], sel_tickers[i]))
    return n_actions,sel_tickers



def get_returns(sel_tickers, df,  minRiskWeights, dateini, ndays, investment=1e5):
    #sel_tickers = list(res_portfolio[1].columns)
    datefin = dateini + BDay(ndays)
    n_actions = get_portfolio_stock_components(minRiskWeights,sel_tickers, df, investment=investment)
    prices_ini = []
    prices_fin = []
    amnt_ini = []
    rho = []
    for i, tick in enumerate(sel_tickers):
        prices_ini.append(df[tick][dateini])
        prices_fin.append(df[tick][datefin])
        amnt_ini.append( df[tick][dateini]*n_actions[i])
        rho.append (df[tick][datefin]*n_actions[i] - df[tick][dateini]*n_actions[i])
    tot_ret = np.array(rho).sum()
    amnt_ini = np.array(amnt_ini).sum()

    return tot_ret/investment*252/ndays-1


def get_returns_tickers(sel_tickers, df,   minRiskWeights, dateini, ndays, investment=1e5):
    #sel_tickers = list(res_portfolio[1].columns)
    datefin = dateini + BDay(ndays)
    n_actions = get_portfolio_stock_components(minRiskWeights,sel_tickers, df, investment=investment)
    prices_ini = []
    prices_fin = []
    amnt_ini = []
    rho = []
    for i, tick in enumerate(sel_tickers):
        prices_ini.append(df[tick][dateini])
        prices_fin.append(df[tick][datefin])
        amnt_ini.append( df[tick][dateini]*n_actions[i])
        rho.append (df[tick][datefin]*n_actions[i] - df[tick][dateini]*n_actions[i])
    tot_ret = np.array(rho).sum()
    amnt_ini = np.array(amnt_ini).sum()

    return tot_ret/investment*252/ndays-1

def get_start_dates(lower_bound, upper_bound, ndays):
    business_days = pd.date_range(start=lower_bound, end=upper_bound, freq=BDay())
    start_dates = np.random.choice(business_days, size=ndays, replace=True)
    return start_dates

def get_random_weights(size = 5):
    # Generate random numbers from a uniform distribution
    random_numbers = np.random.uniform(0, 1, size)
    # Normalize the random numbers to sum up to 1
    normalized_numbers = random_numbers / np.sum(random_numbers)

    return normalized_numbers

def get_random_portfolios(n_port, tickers, df, start_date, final_date):
    rand_port = []
    rand_ret = []
    start_dates = get_start_dates(pd.to_datetime(start_date),pd.to_datetime(final_date), n_port)
    for i in range(n_port):
        randtick = random.sample(tickers, 5)
        wrand = get_random_weights()
        try:
            rho = get_returns_tickers(randtick,df,wrand, pd.to_datetime(start_dates[i]), 5  )
        except:
            pass
        rand_port.append([randtick, wrand, rho ])
        rand_ret.append(rho)
    return rand_port, rand_ret
