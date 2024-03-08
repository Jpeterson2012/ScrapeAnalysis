import numpy as np
import datetime as dt
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
import requests
from sql import get_connection
from util import get_data
import csv

def compute_daily_returns(df):
        daily_returns = df.copy()
        daily_returns = (df / df.shift(1)) - 1
        # daily_returns.iloc[0, :] = 0 
        return daily_returns

def analysis(df):
    # df = df[::-1]
    normed = df/df.iloc[0]
    daily_rets = compute_daily_returns(normed)
    cr, adr, sddr, sr = [
            (normed.iloc[-1]/normed.iloc[0] - 1),
            daily_rets.mean(),
            daily_rets.std(),
            np.sqrt(252.0) * (daily_rets.mean() / daily_rets.std()),
        ]
    print(f"Sharpe Ratio: {sr}")  		  	   		 	   			  		 			     			  	 
    print(f"Volatility (stdev of daily returns): {sddr}")  		  	   		 	   			  		 			     			  	 
    print(f"Average Daily Return: {adr}")  		  	   		 	   			  		 			     			  	 
    print(f"Cumulative Return: {cr}\n\n")

conn, cur = get_connection()

# url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'    

# html_text = requests.get(url)
# soup = BeautifulSoup(html_text.text, 'lxml')
# tickers = soup.find('table')
# tick = tickers.find_all('a', class_='external text')
# syms = []

# for i,t in enumerate(tick):
#     syms.append(t.text)
    # add = (f'insert into SP500 (symbol) values("{syms[i]}")')
    # cur.execute(add)



cur.execute("SELECT symbol FROM SP500")
row = cur.fetchall()

syms = []
for i in range(len(row)):
    syms.append(row[i][0])

syms = syms[0:4]


startDate = dt.datetime(2011, 1, 5)
endDate = dt.datetime(2011, 1, 20)

# df = yf.download(syms, start=startDate, end=endDate)
df = yf.download(syms, period='max', group_by = 'ticker')


N = 6
dfs = np.split(df, np.arange(N, len(df.columns), N), axis=1)

for i in range(len(dfs)):
    dfs[i] = dfs[i].dropna()
    print(dfs[i])
    temp = dfs[i].iloc[:,4]
    analysis(temp)




# df.to_csv('customsyms.csv', index=False)
conn.commit()
cur.close()
conn.close()
