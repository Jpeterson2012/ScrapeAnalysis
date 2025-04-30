import numpy as np
import datetime as dt
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
import requests
from sql import get_connection
import csv

conn, cur = get_connection()

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

cur.execute("SELECT * FROM SP500")
row = cur.fetchall()
syms = []
for i in range(len(row)):
    # syms.append(row[i][0])
    syms.append(row[i][1])

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

def remove_tables():
    cur.execute("SHOW TABLES")
    row = cur.fetchall()
    syms = []
    for i in range(len(row) - 1):
        syms.append(row[i][0])
        cur.execute(f'DROP TABLE `{syms[i]}`')
    print('tables removed successfully')

# remove_tables()


# df.to_csv('customsyms.csv', index=False)
conn.commit()
cur.close()
conn.close()
