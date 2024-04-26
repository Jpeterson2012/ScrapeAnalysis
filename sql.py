import mysql.connector as mysql
import datetime as dt
def get_connection():
    conn = mysql.connect(host = "localhost",user = 'root',password = 'Gearsofwar3!',database = 'finance')
    # conn = mysql.connect(host = "localhost",user = input('Enter user: '),password = input('Enter password: '),database = input('Enter database: '))
    cur = conn.cursor()
    return conn, cur

# startDate = dt.datetime(2011, 1, 5)
# print(int(startDate.strftime("%Y%m%d")))
