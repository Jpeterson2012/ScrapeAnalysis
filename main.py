import datetime as dt
import numpy as np
import pandas as pd
from sql import get_connection
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

#To be used when calling yfinance api
# def split_df(df):
#     N = 6
#     dfs = np.split(df, np.arange(N, len(df.columns), N), axis=1)
#     dfs.reverse()
#     for i in range(len(dfs)):
#         dfs[i] = dfs[i].dropna()
#         print(dfs[i].info())
#         temp = dfs[i].iloc[:,4]
#         analysis(temp)
#     return dfs


#Build dataframe of just Adj Close
stock = 'JPM'
cur.execute(f"SELECT * FROM {stock}")
row = cur.fetchall()

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

analysis(df)

conn.commit()
cur.close()
conn.close()
