#using conda activate scrape
#to run: flask --app [app name name ie index] run
import atexit
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
    return [sr.iloc[0].round(4), sddr.iloc[0].round(4), adr.iloc[0].round(4), cr.iloc[0].round(4)]

def generate_random_percentage():  
    temp = "{:.2f}".format(random.uniform(-10, 10))
    return [ float(temp), temp ]

def topBar():
    sp500 = np.random.permutation(503)[:50].tolist()
    sp500 = ",".join(str(i) for i in sp500)
    
    percents = []
    for i in range(50):        
        percents.append(generate_random_percentage())
    sql = f"SELECT * FROM SP500 WHERE ID IN ({sp500})"
    cur.execute(sql)
    row = cur.fetchall()
    return row, percents

def shutdown():
    conn.commit()
    cur.close()
    conn.close()

atexit.register(shutdown)

@app.route("/")
def index():        
    topbar,percents = topbar2,percents2 = topbar3,percents3 = topbar4,percents4 = topBar()
    return render_template('index.html',data=zip(topbar,percents), data2=zip(topbar2,percents2),data3=zip(topbar3,percents3),data4=zip(topbar4,percents4))

@app.route('/<path:path>')
def catch_all(path):
    return redirect(url_for('index'))
    

@app.route("/stock/<data>")
def stock(data):       
    
    topbar,percents = topBar()

    stock = data
    sql = f"SELECT * FROM {stock} WHERE id BETWEEN 20070806 AND 20121230"
    # sql = f"SELECT * FROM {stock}"
    cur.execute(sql)
    row = cur.fetchall()
    
    for i, r in enumerate(row):
        if 20100104 in r:
            index = i    
    first = row[index:index + 10]
    last = row[-10:]
    rows = []
    rows.extend(first)
    rows.extend(last)   

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
    
    df1.insert(0, 'index_col', df1.index)
    df1['index_col']  = pd.to_datetime(df1['index_col'])
    df1['index_col'] = df1['index_col'].dt.date
    
    rows2 = []
    rows2.extend(df1.head(5).to_numpy().tolist())
    rows2.append(["———","———","———","———","———"])
    rows2.extend(df1.tail(5).to_numpy().tolist())    

    df2.insert(0, 'index_col', df2.index)
    df2['index_col']  = pd.to_datetime(df2['index_col'])
    df2['index_col'] = df2['index_col'].dt.date

    df2[stock] = df2[stock].astype(int)
    df2['Cash'] = df2['Cash'].astype(int)    
    
    rows3 = []
    rows3.extend(df2.to_numpy().tolist())        

    df3 = df3.to_frame()    
    

    df3.insert(0, 'index_col', df3.index)
    df3['index_col']  = pd.to_datetime(df3['index_col'])
    df3['index_col'] = df3['index_col'].dt.date    
    
    
    df3[0] = df3[0].astype(int)
    
    rows4 = []
    rows4.extend(df3.head(5).to_numpy().tolist())
    rows4.append("———")
    rows4.extend(df3.tail(5).to_numpy().tolist())        
           
    metrics = analysis(df)
    
    return render_template('starter.html', message = stock, results = results, data=zip(topbar,percents), tech = rows2, trades = rows3, portvals = rows4, metrics = metrics)

if __name__ == '__main__':
    app.run(debug=True)