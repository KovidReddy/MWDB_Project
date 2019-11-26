<<<<<<< HEAD:task8.py
from dimReduction import dimReduction
from PostgresDB import PostgresDB
arg = input("Please enter the home directory for the images "
            "(Default: C:\\Users\\pylak\\Documents\\Fall 2019\\MWDB\\Project\\Phase1\\Hands_test2\\) : ")
if arg == '':
    arg = 'C:\\Users\\pylak\\Documents\\Fall 2019\\MWDB\\Project\\Phase1\\Hands_test2\\'
dim = dimReduction(arg, '*.jpg')
db = PostgresDB(password='abcdefgh', host='localhost',
                        database='mwdb', user='postgres', port=5432)
conn = db.connect()
bin_matrix, feature_matrix = dim.binMat(conn, 'imagedata_m_pca')
# code to print
# print(bin_matrix[0][0])
dim.imgViz2(bin_matrix)
for idx, sub in enumerate(feature_matrix):
    print('\nLatent Semantic {x}'.format(x=idx+1))
    for s in sub:
        print('Subject: ', s[0])
        print('Weight: ', s[1])
=======
from dimReduction import dimReduction
from PostgresDB import PostgresDB
arg = input("Please enter the home directory for the images "
            "(Default: C:\\Users\\pylak\\Documents\\Fall 2019\\MWDB\\Project\\Phase1\\Dataset\\) : ")
if arg == '':
    arg = 'C:\\Users\\pylak\\Documents\\Fall 2019\\MWDB\\Project\\Phase1\\Dataset\\'
dim = dimReduction(arg, '*.jpg')
db = PostgresDB(password='1Idontunderstand', host='localhost',
                        database='postgres', user='postgres', port=5432)
conn = db.connect()
# Create for M and PCA
_, _, _, _ = dim.saveDim('m', 'pca', 'imagedata_m', 10, password ="1Idontunderstand", database="postgres")
bin_matrix, feature_matrix = dim.binMat(conn, 'imagedata_m_pca')
# code to print
# print(bin_matrix[0][0])
dim.imgViz(bin_matrix)
for idx, sub in enumerate(feature_matrix):
    print('\nLatent Semantic {x}'.format(x=idx+1))
    for s in sub:
        print('Subject: ', s[0])
        print('Weight: ', s[1])
>>>>>>> origin/Kovid_branch:BACKUP/task8.py
