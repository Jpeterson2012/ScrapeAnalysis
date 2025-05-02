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
    print(f"Sharpe Ratio: {sr}")  		  	   		 	   			  		 			     			  	 
    print(f"Volatility (stdev of daily returns): {sddr}")  		  	   		 	   			  		 			     			  	 
    print(f"Average Daily Return: {adr}")  		  	   		 	   			  		 			     			  	 
    print(f"Cumulative Return: {cr}\n\n")

def generate_random_percentage():  
    temp = "{:.2f}".format(random.uniform(-10, 10))
    return [ float(temp), temp ]

@app.route("/")
def hello_world():        

    sp500 = np.random.permutation(503)[:20].tolist()
    sp500 = ",".join(str(i) for i in sp500)
    
    percents = []
    for i in range(20):        
        percents.append(generate_random_percentage())
    sql = f"SELECT * FROM SP500 WHERE ID IN ({sp500})"
    cur.execute(sql)
    row = cur.fetchall()
    topbar = row

    stock = 'GOOG'
    cur.execute(f"SELECT * FROM {stock}")
    row = cur.fetchall()

    # sym = []
    # d_index = []
    # for i in range(len(row)):
    #     d_index.append(dt.datetime.strptime(str(row[i][0]), '%Y%m%d'))
    #     sym.append(round(float(row[i][4]),2))
    # df = pd.DataFrame(data=sym, index=d_index, columns=[stock])

    results = []    
    for i in range(10):                
        results.append([ dt.datetime.strptime(str(row[i][0]), '%Y%m%d').date(), "{:.3f}".format(row[i][1]), "{:.3f}".format(row[i][2]), "{:.3f}".format(row[i][3]), 
        "{:.3f}".format(row[i][4]), "{:.3f}".format(row[i][5]), row[i][6] ])
            
    sym = []
    d_index = []
    # for i in range(len(row)):
    #     d_index.append(dt.datetime.strptime(str(row[i][0]), '%Y%m%d'))
    #     sym.append(round(float(row[i][4]),2))
    # df = pd.DataFrame(data=sym, index=d_index, columns=[stock])

    # #Call ml model on dataframe
    # a = sl.StrategyLearner()
    # a.add_evidence(symbol=stock,dframe=df)
    # a.testPolicy(symbol=stock,dframe=df)

    # analysis(df)

    conn.commit()
    cur.close()
    conn.close()

    return render_template('index.html', message = stock, results = results, data=zip(topbar,percents))

