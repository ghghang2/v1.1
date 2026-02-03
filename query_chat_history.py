import sqlite3
import json

conn = sqlite3.connect('chat_history.db')
cursor = conn.cursor()
# fetch schema
cursor.execute("PRAGMA table_info(chat_history);")
info = cursor.fetchall()
print('Schema:', info)
# check if any column contains 'reasoning'
for row in info:
    if 'reasoning' in row[1].lower():
        print('Found column:', row[1])
# fetch rows with reasoning content
cursor.execute("SELECT id, reasoning FROM chat_history;")
rows = cursor.fetchall()
print('Rows with reasoning:', rows[:5])

conn.close()
