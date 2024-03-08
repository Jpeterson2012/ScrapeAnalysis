import mysql.connector as mysql
def get_connection():
    conn = mysql.connect(host = "localhost",user = input('Enter user: '),password = input('Enter password: '),database = input('Enter database: '))
    cur = conn.cursor()
    return conn, cur
