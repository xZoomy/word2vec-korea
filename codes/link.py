import sqlite3

db = sqlite3.connect('./database/kernel.sqlite')
c = db.cursor()

query = """select assembly from functions limit 5;"""
print(c.execute(query))
c.close()
