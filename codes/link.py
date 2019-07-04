import sqlite3

db = sqlite3.connect('../databases/kernel.sqlite')
c = db.cursor()

query = """select assembly from functions limit 5;"""
c.execute(query)
rows = c.fetchall()

for row in rows:
    print(row)

c.close()
