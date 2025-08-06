import sqlite3

dbPath = ""

def openConneciton():
    sqliteDb = sqlite3.connect(dbPath)
    return sqliteDb

def execCommand(query, params, returnIndetity = False):
    id = 0
    db = openConneciton()
    cursor = db.cursor()
    if params is None:
        cursor.execute(query)
    else:
        cursor.execute(query, params)
    if returnIndetity:
        cursor.execute("SELECT last_insert_rowid() AS id")
        rows = cursor.fetchall()
        id = rows[0][0]
    db.commit()
    db.close()
    return id

def getData(query, params):
    db = openConneciton()
    cursor = db.cursor()
    if(params is not None and len(params) > 0):
        cursor.execute(query,params)
    else:
        cursor.execute(query)
    rows = cursor.fetchall()
    db.close()
    return  rows
