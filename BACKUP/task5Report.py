from dimReduction import dimReduction
from PostgresDB import PostgresDB
import os
path = input("Please enter the home directory for the images "
            "(Default: C:\\Users\\pylak\\Documents\\Fall 2019\\MWDB\\Project\\Phase1\\Hands_test2\\) : ")
if path == '':
    path = 'C:\\Users\\pylak\\Documents\\Fall 2019\\MWDB\\Project\\Phase1\\Dataset2\\'
dim = dimReduction(path, '*.jpg')
db = PostgresDB(password='abcdefgh', host='localhost',
                        database='mwdb', user='postgres', port=5432)
conn = db.connect()
cur = conn.cursor()
cur.execute("SELECT image_id FROM imagedata_m INNER JOIN img_meta t on t.image_id = imageid WHERE t.orient = 'right'")
male = [x[0] for x in cur.fetchall()][0:15]
cur.execute("SELECT image_id FROM imagedata_m INNER JOIN img_meta t on t.image_id = imageid WHERE t.orient = 'left'")
female = [x[0] for x in cur.fetchall()][0:15]
tot = len(male) + len(female)
print(tot)
cnt = 0
for feature in ['s']:
    for tech in ['pca', 'nmf', 'lda', 'svd']:
        print(feature, tech)
        for m in male:
            pred = dim.classifyImg(conn, feature, m, 'right', tech)
            if pred == 1:
                cnt += 1
        for f in female:
            pred = dim.classifyImg(conn, feature, f, 'right', tech)
            if pred == -1:
                cnt += 1
        print('Acc', float(cnt/tot) * 100)
        cnt = 0
        print('\n')