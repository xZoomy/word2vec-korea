#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 14:46:36 2019

@author: jlphung
"""

import sqlite3
from sqlite3 import Error


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return None

def select_assembly(conn):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT assembly || char(10) FROM functions;")
    rows = cur.fetchall()
    f=open("../input/assembly.asm","w+")
    for row in rows:
        for i in range(0,len(row)):
            f.write(row[i])
        #print(type(row))
    print("input/assembly.asm created")
    f.close()




def main():
    database = "../databases/kernel.sqlite"
    conn = create_connection(database)
    with conn:
        select_assembly(conn)


if __name__ == '__main__':
    main()
