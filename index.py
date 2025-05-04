#using conda activate scrape
#to run: flask --app display run

from flask import Flask, render_template, request, jsonify
import datetime as dt
import numpy as np  	   		 	   			  		 			     			  	   	   		 	   			  		 			     			  	   		  	   		 	   			  		 			     			  	   		  	   		 	   			  		 			     			  	   		  	   		 	   			  		 	
import pandas as pd
from sql import get_connection, myCors
import random
import StrategyLearner as sl



app = Flask(__name__)
cors = myCors(app)
app.config['CORS_HEADERS'] = 'Content-Type'

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
    # print(f"Sharpe Ratio: {sr}")  		  	   		 	   			  		 			     			  	 
    # print(f"Volatility (stdev of daily returns): {sddr}")  		  	   		 	   			  		 			     			  	 
    # print(f"Average Daily Return: {adr}")  		  	   		 	   			  		 			     			  	 
    # print(f"Cumulative Return: {cr}\n\n")    
    return [sr.iloc[0], sddr.iloc[0], adr.iloc[0], cr.iloc[0]]

def generate_random_percentage():  
    temp = "{:.2f}".format(random.uniform(-10, 10))
    return [ float(temp), temp ]

@app.route("/")
def hello_world():        

    sp500 = np.random.permutation(503)[:35].tolist()
    sp500 = ",".join(str(i) for i in sp500)
    
    percents = []
    for i in range(35):        
        percents.append(generate_random_percentage())
    sql = f"SELECT * FROM SP500 WHERE ID IN ({sp500})"
    cur.execute(sql)
    row = cur.fetchall()
    topbar = row

    stock = 'JPM'
    cur.execute(f"SELECT * FROM {stock}")
    row = cur.fetchall()
    first = row[:10]
    last = row[-10:]
    rows = []
    rows.extend(first)
    rows.extend(last)

    # sym = []
    # d_index = []
    # for i in range(len(row)):
    #     d_index.append(dt.datetime.strptime(str(row[i][0]), '%Y%m%d'))
    #     sym.append(round(float(row[i][4]),2))
    # df = pd.DataFrame(data=sym, index=d_index, columns=[stock])

    results = []    
    for i in range(len(rows)):                
        results.append([ dt.datetime.strptime(str(rows[i][0]), '%Y%m%d').date(), "{:.2f}".format(rows[i][1]), "{:.2f}".format(rows[i][2]), "{:.2f}".format(rows[i][3]), 
        "{:.2f}".format(rows[i][4]), "{:.2f}".format(rows[i][5]), "{:,}".format(rows[i][6]) ])
        if (i == 9):
            results.append(["———","———","———","———","———","———","———"])

            
    sym = []
    d_index = []
    for i in range(len(row)):
        d_index.append(dt.datetime.strptime(str(row[i][0]), '%Y%m%d'))
        sym.append(round(float(row[i][4]),2))
    df = pd.DataFrame(data=sym, index=d_index, columns=[stock])

    #Call ml model on dataframe
    a = sl.StrategyLearner()
    a.add_evidence(symbol=stock,dframe=df)
    df1,df2, df3 = a.testPolicy(symbol=stock,dframe=df)
    df1.columns = pd.RangeIndex(df1.shape[1]) #using RangeIndex

    # print(df2)        

    rows2 = []
    rows2.extend(df1.head(5).to_numpy().tolist())
    rows2.append(["———","———","———","———","———"])
    rows2.extend(df1.tail(5).to_numpy().tolist())    

    df2.insert(0, 'index_col', df2.index)
    df2['index_col']  = pd.to_datetime(df2['index_col'])
    df2['index_col'] = df2['index_col'].dt.date
    
    rows3 = []
    rows3.extend(df2.to_numpy().tolist())        

    rows4 = []
    rows4.extend(df3.head(5))
    rows4.append("———")
    rows4.extend(df3.tail(5))    
       
    

    metrics = analysis(df)

    conn.commit()
    cur.close()
    conn.close()

    return render_template('index.html', message = stock, results = results, data=zip(topbar,percents), tech = rows2, trades = rows3, portvals = rows4, metrics = metrics)

