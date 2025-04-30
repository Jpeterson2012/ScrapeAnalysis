import numpy as np
import datetime as dt
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
import requests
from sql import get_connection
import csv
import StrategyLearner as sl

conn, cur = get_connection()

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

def scrape():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'    

    html_text = requests.get(url)
    soup = BeautifulSoup(html_text.text, 'lxml')
    tickers = soup.find('table')
    tick = tickers.find_all('a', class_='external text')
    syms = []
    for i,t in enumerate(tick):
        syms.append(t.text)
        add = (f'insert into SP500 (symbol) values("{syms[i]}")')
        cur.execute(add)

def remove_tables():
    cur.execute("SHOW TABLES")
    row = cur.fetchall()
    syms = []
    for i in range(len(row) - 1):
        syms.append(row[i][0])
        cur.execute(f'DROP TABLE `{syms[i]}`')
    print('tables removed successfully')

# remove_tables()
# cur.execute("SELECT * FROM SP500")
# cur.execute("SELECT * FROM SP500 WHERE symbol = 'ZION'")
# row = cur.fetchall()
# syms = []
# for i in range(len(row)):
#     # syms.append(row[i][0])
#     syms.append(row[i][1])

stock = 'JPM'
cur.execute(f"SELECT * FROM {stock}")
row = cur.fetchall()

#Build dataframe of just Adj Close
sym = []
d_index = []
for i in range(len(row)):
    d_index.append(dt.datetime.strptime(str(row[i][0]), '%Y%m%d'))
    sym.append(round(float(row[i][4]),2))
df = pd.DataFrame(data=sym, index=d_index, columns=[stock])

#Call ml model on dataframe
a = sl.StrategyLearner()
a.add_evidence(symbol=stock,dframe=df)
a.testPolicy(symbol=stock,dframe=df)

#Build dataframe of historical data for stock
# sym = []
# d_index = []
# for i in range(len(row)):
#     temp = []
#     for j in range(7):
#         if j == 0:
#             d_index.append(dt.datetime.strptime(str(row[i][j]), '%Y%m%d'))
#         else:
#             temp.append(row[i][j])
#     sym.append(temp)
# print(sym[0])
# # print(d_index[0:4])
# # print(dt.datetime.strptime(str(sym[0][0]), '%Y%m%d'))

# df = pd.DataFrame(data=sym, index=d_index, columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])



startDate = dt.datetime(2011, 1, 5)
endDate = dt.datetime(2011, 1, 20)

#Custom date range
# start_index = df.index.get_loc(startDate)
# end_index = df.index.get_loc(endDate) + 1
# print(df[start_index:end_index])

# df = yf.download(syms, start=startDate, end=endDate)
# df = yf.download(syms, period='max', group_by = 'ticker')
# df = yf.download(syms, period='max')
# print(df)
def build_tables():
    for i in range(len(syms)):
        df = yf.download(syms[i], period='max')
        df = df.dropna()
        data = f'''CREATE TABLE `{syms[i]}` (
            id INT PRIMARY KEY,
            Open DECIMAL(14,10),
            High DECIMAL(14,10),
            Low DECIMAL(14,10),
            Close DECIMAL(14,10),
            Adj_Close DECIMAL(14,10),
            Volume BIGINT(15)
        );'''
        cur.execute(data)
        for j in range(df.shape[0]):
            add = (f'insert into `{syms[i]}` values(  {int(df.index[j].strftime("%Y%m%d"))}, {df.iloc[j,0]}, {df.iloc[j,1]}, {df.iloc[j,2]}, {df.iloc[j,3]}, {df.iloc[j,4]}, {df.iloc[j,5]}   )')
            cur.execute(add)

# build_tables()


#To be used when calling yfinance api
def split_df(df):
    N = 6
    dfs = np.split(df, np.arange(N, len(df.columns), N), axis=1)
    dfs.reverse()
    for i in range(len(dfs)):
        dfs[i] = dfs[i].dropna()
        print(dfs[i].info())
        temp = dfs[i].iloc[:,4]
        analysis(temp)
    return dfs

analysis(df)

# df.to_csv('customsyms.csv', index=False)
conn.commit()
cur.close()
conn.close()
