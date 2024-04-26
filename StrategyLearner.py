""""""  		  	   		 	   			  		 			     			  	 
"""  		  	   		 	   			  		 			     			  	 
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	   			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		 	   			  		 			     			  	 
All Rights Reserved  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Template code for CS 4646/7646  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	   			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		 	   			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		 	   			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		 	   			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	   			  		 			     			  	 
or edited.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		 	   			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		 	   			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	   			  		 			     			  	 
GT honor code violation.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
-----do not edit anything above this line---  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Student Name: Jesse Peterson	  	   		 	   			  		 			     			  	 
GT User ID: jpeterson93 		  	   		 	   			  		 			     			  	 
GT ID: 903949363  		  	   		 	   			  		 			     			  	 
"""  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import datetime as dt
from datetime import timedelta  		  	   		 	   			  		 			     			  	 
import random
import time  		  	   		 	   			  		 			     			  	 

import numpy as np               		 	   			  		 			     			  	 
import pandas as pd  	
import matplotlib.pyplot as plt	  	   		 	   			  		 			     			  	 
from util import get_data
import RTLearner as rt	
import BagLearner as bt

def compute_daily_returns(df):
    daily_returns = df.copy()
    daily_returns = (df / df.shift(1)) - 1
    daily_returns.iloc[0, :] = 0 
    return daily_returns

def bbands(prices):
    lookback = 80
    # lookback = 100
    sma = prices.rolling( window=lookback, min_periods=lookback).mean()
    std = prices.rolling( window=lookback, min_periods=lookback).std()
    upper_band = sma + std * 2
    lower_band = sma - std * 2
    #Vectorize
    bbp = (prices - lower_band) / (upper_band - lower_band)
    return bbp

def price_sma_ratio(prices):
    lookback = 35
    # lookback = 20
    # lookback = 100
    # price = prices[11:]
    price = prices
    sma = price.rolling( window=lookback, min_periods=lookback).mean()
    psma = price / sma
    return psma

def momentum(prices):
    lookback = 50
    # lookback = 20
    # lookback = 100
    mmtum = prices.copy()
    mmtum.iloc[0:lookback,0] = np.nan
    for day in range(lookback,prices.shape[0]):
        mmtum.iloc[day,0] = (prices.iloc[day,0] / prices.iloc[day-lookback,0]) - 1
    return mmtum

def rsi(prices):
    # price = prices[5:]
    price = prices
    daily_rets = compute_daily_returns(price)
    rsi = price.copy()
    lookback = 14
    # lookback = 25

    for day in range(price.shape[0]):
        if day <= lookback:
            rsi.iloc[day, 0] = np.nan
            continue

        up_gain = daily_rets.iloc[day-lookback+1:day+1,:].where(daily_rets >= 0).sum()
        down_loss = -1 * daily_rets.iloc[day-lookback+1:day+1,:].where(daily_rets < 0).sum()
        rs = (up_gain/lookback) / (down_loss / lookback)
        rsi.iloc[day,:] = 100 - (100 / (1 + rs))
    rsi[rsi == np.inf] = 100
    return rsi

def buySell(price,psratio,df):
    dates1 = []
    dates2 = []
    last_trade = 0
    no_trade = 0
    for i in range(1,df.shape[0]):

        if no_trade > i:
            pass
        else:
            if (df.iloc[i,0] < 0 and df.iloc[i,1] < 1 and df.iloc[i,2] < 0 and df.iloc[i,3] < 50):
                # print(df.index[i])
                
                if last_trade == 0:    
                    df.iloc[i,-1] = 1
                    last_trade = price.iloc[i,0]
                    dates1.append(df.index[i])
                else:
                    if price.iloc[i,0] > last_trade * 1.0 or price.iloc[i,0] < last_trade * 0.99:
                        df.iloc[i,-1] = 1
                        last_trade = price.iloc[i,0]
                        dates1.append(df.index[i])
                        # no_trade = i + 30
                        # if no_trade > df.shape[0]:
                        #     break

            if (df.iloc[i,0] > 0.5 and df.iloc[i,1] > 1 and df.iloc[i,2] > 0 and df.iloc[i,3] > 50):
                # print(df.index[i])
                
                if last_trade == 0:
                    df.iloc[i,-1] = -1
                    last_trade = price.iloc[i,0]
                    dates2.append(df.index[i])
                else:
                    if price.iloc[i,0] > last_trade * 1.0 or price.iloc[i,0] < last_trade * 0.99:
                        df.iloc[i,-1] = -1
                        last_trade = price.iloc[i,0]
                        dates2.append(df.index[i])
                        # no_trade = i + 30
                        # if no_trade > df.shape[0]:
                        #     break
    return dates1, dates2


def plot_func(p, i, t, l1=0,l2=0, l3=['2008-01-02', '2009-01-02'], l4=['2008-03-02', '2009-03-02']):
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price', color=color)
    ax1.plot(p, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(t, color=color)  # we already handled the x-label with ax1
    ax2.plot(i, color=color)
    ax2.axhline(y = l1)
    ax2.axhline(y = l2)
    # ax2.axvline(x = i.index[80])
    for i in range(len(l3)):
        ax2.axvline(x = l3[i], color='r')
    for i in range(len(l4)):
        ax2.axvline(x = l4[i], color = 'g')
    
    # ax2.axvline(x = i.index[130])
    # ax2.axvline(x = i.index[180])
    # ax2.axvline(x = i.index[230])
    # ax2.axvline(x = i.index[280])
    # ax2.axvline(x = i.index[330])
    
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()       
        
  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
class StrategyLearner(object):  		  	   		 	   			  		 			     			  	 
      	   		 	   			  		 			     			  	 
    # constructor  		  	   		 	   			  		 			     			  	 
    def __init__(self, verbose=False, impact=0.0, commission=0.0):  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
        Constructor method  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
        self.verbose = verbose  		  	   		 	   			  		 			     			  	 
        self.impact = impact  		  	   		 	   			  		 			     			  	 
        self.commission = commission
        
        self.learner = {}
        self.bband = {}
        self.psratio = {}
        self.mtum = {}
        self.rsii = {}  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    # this method should create a QLearner, and train it for trading  		  	   		 	   			  		 			     			  	 
    def add_evidence(  		  	   		 	   			  		 			     			  	 
        self,  		  	   		 	   			  		 			     			  	 
        symbol,
        dframe,  		  	   		 	   			  		 			     			  	 
        sd=dt.datetime(2008, 1, 3),  		  	   		 	   			  		 			     			  	 
        ed=dt.datetime(2009, 12, 30),  		  	   		 	   			  		 			     			  	 
        sv=100000,  		  	   		 	   			  		 			     			  	 
    ):  		  	   		 	   			  		 			     			  	 		 	   			  		 			     			  	 		  	   		 	   			  		 			     			  	 

        # syms = [symbol]
        start = sd - timedelta(days=150)
        end = dt.datetime(2011,12,30)
        # prices = get_data(syms, pd.date_range(sd, ed))
        # prices2 = get_data(syms, pd.date_range(start, end))
        # prices = prices[syms]
        # prices2 = prices2[syms]
        
        start_index = dframe.index.get_loc(sd)
        end_index = dframe.index.get_loc(ed) + 1
        prices = dframe[start_index:end_index]

        start_index = dframe.index.get_loc(start)
        end_index = dframe.index.get_loc(end) + 1
        prices2 = dframe[start_index:end_index]

        
        start_index = prices2.index.get_loc(prices.index[0])
        end_index = prices2.index.get_loc(prices.index[-1]) + 1
        
        
        bband = bbands(prices2)
        self.bband = bband.copy()
        bband = bband[symbol].iloc[start_index:end_index]

        psratio = price_sma_ratio(prices2)
        self.psratio = psratio.copy()
        psratio = psratio[symbol].iloc[start_index:end_index]

        mtum = momentum(prices2)
        self.mtum = mtum.copy()
        mtum = mtum[symbol].iloc[start_index:end_index]

        rsii = rsi(prices2)
        self.rsii = rsii.copy()
        rsii = rsii[symbol].iloc[start_index:end_index]

        d = {'BBANDS': bband, 'PSRATIO': psratio, 'Momentum': mtum, 'RSI': rsii}
        df = pd.DataFrame(data = d, index = bband.index)
        # df.to_csv('indicators.csv', index=False)
        df.insert(df.shape[1], 'Signal', [0 for i in range(df.shape[0])])
        
        d1, d2 = buySell(prices,psratio,df)

        train_x = df.iloc[:, :-1]
        train_y = df.iloc[:, -1]

        self.learner = bt.BagLearner(learner=rt.RTLearner, kwargs={'leaf_size':5},bags = 20)
        self.learner.add_evidence(train_x, train_y)

        plot_func(prices[symbol], bband, 'BBands', 0, 1, d1, d2)
        # plot_func(prices[symbol], psratio, 'PSRATIO', 1, 1,d1,d2)
        # plot_func(prices[symbol], mtum, 'Momentum', 0, 0,d1,d2)
        # plot_func(prices[symbol], rsii, 'RSI', 30, 70,d1,d2)
 		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    # this method should use the existing policy and test it against new data  		  	   		 	   			  		 			     			  	 
    def testPolicy(  		  	   		 	   			  		 			     			  	 
        self,  		  	   		 	   			  		 			     			  	 
        symbol,
        dframe,  		  	   		 	   			  		 			     			  	 
        sd=dt.datetime(2010, 1, 4),  		  	   		 	   			  		 			     			  	 
        ed=dt.datetime(2011, 12, 30),  		  	   		 	   			  		 			     			  	 
        sv=100000,  		  	   		 	   			  		 			     			  	 
    ):  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
        Tests your learner using data outside of the training data  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
        :param symbol: The stock symbol that you trained on on  		  	   		 	   			  		 			     			  	 
        :type symbol: str  		  	   		 	   			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	   			  		 			     			  	 
        :type sd: datetime  		  	   		 	   			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	   			  		 			     			  	 
        :type ed: datetime  		  	   		 	   			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		 	   			  		 			     			  	 
        :type sv: int  		  	   		 	   			  		 			     			  	 
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		 	   			  		 			     			  	 
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		 	   			  		 			     			  	 
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		 	   			  		 			     			  	 
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		 	   			  		 			     			  	 
        :rtype: pandas.DataFrame  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
        # syms = [symbol]
        # prices = get_data(syms, pd.date_range(sd, ed))
        # prices = prices[syms]

        start_index = dframe.index.get_loc(sd)
        end_index = dframe.index.get_loc(ed) + 1
        prices = dframe[start_index:end_index]
        prices.insert(prices.shape[1], 'Cash', [1. for i in range(prices.shape[0])])
        

        bb = self.bband
        start_index = bb.index.get_loc(prices.index[0])
        end_index = bb.index.get_loc(prices.index[-1]) + 1

        bb = bb[symbol].iloc[start_index:end_index]
        psr = self.psratio
        psr = psr[symbol].iloc[start_index:end_index]
        mm = self.mtum
        mm = mm[symbol].iloc[start_index:end_index]
        rs = self.rsii
        rs = rs[symbol].iloc[start_index:end_index]

        
        d = {'BBANDS': bb, 'PSRATIO': psr, 'Momentum': mm, 'RSI': rs}
        df = pd.DataFrame(data = d, index = bb.index)
        #Generate y values through bag learner
        # start = time.time()
        yvals = self.learner.query(df)
        # end = time.time()
        # print(f'Time: {end - start}')
        df.insert(df.shape[1], 'Signal', yvals.T)
        print(df)

        trades = prices.copy()
        for col in trades.columns:
            trades[col] = 0.
        holdings = trades.copy()
        values = trades.copy()
        holdings.iloc[0,-1] = sv

        
        current_trade = 0.
        for i in range(1,len(trades.index) - 1):
            if df.iloc[i-1,-1] == 1.0:
                impactt = (1.0 + self.impact)
                if current_trade == -1000.0:
                    trades.iloc[i,0] = 2000.0
                    trades.iloc[i,1] = ((trades.iloc[i,0] * prices.iloc[i, 0] * impactt) + self.commission) * -1
                elif current_trade == 0.0:
                    trades.iloc[i,0] = 1000.0
                    trades.iloc[i,1] = ((trades.iloc[i,0] * prices.iloc[i, 0] * impactt) + self.commission) * -1
                else:
                    pass
                current_trade += trades.iloc[i,0]
            if df.iloc[i-1,-1] == -1.0:
                impactt = (1.0 - self.impact)
                if current_trade == 1000.0:
                    trades.iloc[i,0] = -2000.0
                    trades.iloc[i,1] = ((trades.iloc[i,0] * prices.iloc[i, 0] * impactt) - self.commission) * -1
                elif current_trade == 0.0:
                    trades.iloc[i,0] = -1000.0
                    trades.iloc[i,1] = ((trades.iloc[i,0] * prices.iloc[i, 0] * impactt) - self.commission) * -1
                else:
                    pass
                current_trade += trades.iloc[i,0]
            
            

        for i in range(holdings.shape[0]):
            if i == 0:
                holdings.iloc[i, 0:-1] += trades.iloc[i, 0:-1]
                holdings.iloc[i, -1] += (trades.iloc[i, -1])
            else:
                holdings.iloc[i, 0:-1] = holdings.iloc[i-1, 0:-1] + trades.iloc[i, 0:-1]
                holdings.iloc[i, -1] = holdings.iloc[i-1, -1] + (trades.iloc[i, -1])
            
            values.iloc[i, 0:-1] = prices.iloc[i, 0:-1] * holdings.iloc[i, 0:-1]
            values.iloc[i, -1] = holdings.iloc[i, -1]
        
        portvals = (values.sum(axis=1))
        
    
        
        
        trades = trades[trades.iloc[:,0]!= 0.0]
        
        print(trades)
        # trade = pd.DataFrame(data = trades[symbol])
        print(portvals.tail())
        
        return pd.DataFrame(data = trades[symbol])

        
                                                                                                                                                                                               
        # return portvals


  	   		 	   			  		 			     			  	 
