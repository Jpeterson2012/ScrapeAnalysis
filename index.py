#using conda activate scrape
#to run: flask --app display run

from flask import Flask, render_template, request, jsonify
import datetime as dt
import numpy as np  	   		 	   			  		 			     			  	   	   		 	   			  		 			     			  	   		  	   		 	   			  		 			     			  	   		  	   		 	   			  		 			     			  	   		  	   		 	   			  		 	
import pandas as pd
from sql import get_connection, myCors

app = Flask(__name__)
cors = myCors(app)
app.config['CORS_HEADERS'] = 'Content-Type'

conn, cur = get_connection()

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




app = Flask(__name__)

# @app.route("/getStuff/", methods=["POST"])
# def stuff():
#     # message = request.form.get("data")
#     temp = request.form.get("data").splitlines()
#     temp2 = request.form.get("data")
#     arr = temp2.splitlines()
#     newarr = [f'"{item}"' for item in arr]
#     temp = ','.join(newarr)
#     sql = f'SELECT * FROM ipads WHERE id IN ({temp})'
#     mycursor.execute(sql)
#     results = mycursor.fetchall()
#     message = len(results)

#     return render_template('starter.html', message = message, results = results)

@app.route("/")
def hello_world():    

    # bcodes = "F9FRP0NFGHMK,DLXQF0LPGHMK,DLXSG1BHHGC4,F9FQC232GHMK,DLXR60KXGHMG,F9FQK62YGHMG,F9FTM1DLGHMN,F9FQM1YAGHMG,DLXSD4YYHGC4,DLXQP0PNGHML,F9FVL3SPGHMN,F9FVW19XGHMN,F9FTK35TGHMN,F9FQ74M9GHMK,F9FT62PLGHMN,DLXSL14DHGC3,F9FS88K9GHMQ,F9FV33QSGHMP,F9FVV5NPGHMN,F9FWV4MFGHMN,F9FT53U0HGC4,F9FSK1KHHGC4,F9FTG45CGHMN,DLXR41DLGHMG,DLXR4463GHMG,F9FT51PCGHMN,F9FT91VFHGC3,DLXRC1Q2GHMK,DLXQG1QGGHMG,DLXS40VWGHMG,DLXSK18JHGC3,FNXDK00BGHMH,DLXQC1E4GHMM,FFKW201VGHMH,F9FQN14TGHMG,FFKV300MHGC5,DLXSDBXQHGC5,F9FXR0NJGHMN,DLXRC3LCGHMJ,DLXQ70KVGHML,FFKYC009GHMH,FFLTQ001GHMH,FFLDN008HGC5,FFLYJ079GHMN,F9FW8916GHMN,FFLW800CHGC4,F9FW56CHGHMN,FFKW302EHGC5,F9FRV14QGHMJ,F9FS809XGHML,DMPYQXEXJF88,DMPTL2UTHLJJ,GCGV8ZC3HLJJ,GCGV8W9XHLJJ,F9GV96V2HLJJ,DMPV145KHLJJ,GG7W84S7HLJJ,GCGV990CHLJJ,GCGV97U3HLJJ,DMPV11HRHLJJ,GG7W6A97HLJJ,GG7YH4H3JF88,GCTW45XYHLJL,F9FTLMPQHLJJ,F9FTVNBHHLJJ,DMPYJD50JF88,DMPX8HPCJF88,GG7YH93AJF88,GG7YJ75RJF88,DMPYKJDCJF88,GCTVTGDZHLJJ,GCTW25A5HLJJ,DMPYQSX8JF88,GG7YJD3FJF88,DMPYQUGNJF88,GCTW2MBBHLJJ,DMPYJ7SYJF88,GG7YJ5HVJF88"
    # arr = bcodes.split("\n")
    # newarr = [f'"{item}"' for item in arr]
    # temp = ','.join(newarr)
    # sql = f'SELECT * FROM ipads WHERE id IN ({temp})'
    # # mycursor.execute('Select * from ipads where id in ("DMPWWLC9JMXJ,GG7YJ383JF88") ')
    # mycursor.execute(sql)
    # results = mycursor.fetchall()

    stock = 'JPM'
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
        results.append([ dt.datetime.strptime(str(row[i][0]), '%Y%m%d').date(), round(float(row[i][1]), 3), round(float(row[i][2]), 3), round(float(row[i][3]), 3), 
        round(float(row[i][4]), 3), round(float(row[i][5]), 3), row[i][6] ])
    
    print(results)
    message = stock
    

    return render_template('index.html', message = message, results = results)

# @app.route("/api/barcodes")
# @cross_origin()
# def oof():
#     bcodes = "F9FRP0NFGHMK,DMPYKJDCJF88,GCTVTGDZHLJJ,GCTW25A5HLJJ,DMPYQSX8JF88,GG7YJD3FJF88,DMPYQUGNJF88,GCTW2MBBHLJJ,DMPYJ7SYJF88,GG7YJ5HVJF88"
#     arr = bcodes.split(",")
#     newarr = [f'"{item}"' for item in arr]
#     temp = ','.join(newarr)
#     sql = f'SELECT * FROM ipads WHERE id IN ({temp})'
#     # mycursor.execute('Select * from ipads where id in ("DMPWWLC9JMXJ,GG7YJ383JF88") ')
#     mycursor.execute(sql)
#     results = mycursor.fetchall()
#     # for i in range(0,len(results)):
#     #   print(results)
    
#     return jsonify({"results": results})
