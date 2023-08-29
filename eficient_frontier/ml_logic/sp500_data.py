import pandas as pd
from yahoo_fin.stock_info import get_data
import numpy as np
from tqdm import tqdm
import pickle as pkl
from datetime import datetime, timedelta


def update_sp500_data():

    constituents_df = pd.read_csv('/home/mina/code/Xryrapier/eficient_frontier/raw_data/constituents_csv.csv')
    today = datetime.now().date()
    start_date = '2010-01-01'
    end_date = today
    ladj = []
    tickers=[]
    for ticker in tqdm(list(constituents_df['Symbol'])):
        try:
            df2 = get_data(ticker, start_date=start_date, end_date=end_date, index_as_date = True)
            ladj.append(df2)
            tickers.append(ticker)
        except:
            print(ticker, 'not found')

    with open('sp500_all.pkl', 'wb') as f:
        pkl.dump([tickers, ladj], f)

def get_sp500_data ():
    with open('/home/mina/code/Xryrapier/eficient_frontier/raw_data/sp500_all.pkl'
, 'rb') as f:
        sp500_all = pkl.load(f)
    return sp500_all[0], sp500_all[1]
