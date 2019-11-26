<<<<<<< HEAD:task6.py
from dimReduction import dimReduction
from PostgresDB import PostgresDB
arg = input("Please enter the home directory for the images "
            "(Default: C:\\Users\\pylak\\Documents\\Fall 2019\\MWDB\\Project\\Phase1\\Hands_test2\\) : ")
if arg == '':
    arg = 'C:\\Users\\pylak\\Documents\\Fall 2019\\MWDB\\Project\\Phase1\\Hands_test2\\'
dim = dimReduction(arg, '*.jpg')
arg1 = input("Please enter the Subject ID you would want to compare: ")
db = PostgresDB(password='abcdefgh', host='localhost',
                        database='mwdb', user='postgres', port=5432)
conn = db.connect()
# Change the database name in case you want to test with a different combination of features and dim
sub_matrix = dim.subMatrix(conn, 'imagedata_l_svd', arg1, mat=False)
for sub in sub_matrix:
    print('Subject: ', sub[0])
    print('Weight:', sub[1])
    print('\n')
=======
from dimReduction import dimReduction
from PostgresDB import PostgresDB
arg = input("Please enter the home directory for the images "
            "(Default: C:\\Users\\pylak\\Documents\\Fall 2019\\MWDB\\Project\\Phase1\\Dataset\\) : ")
if arg == '':
    arg = 'C:\\Users\\pylak\\Documents\\Fall 2019\\MWDB\\Project\\Phase1\\Dataset\\'
dim = dimReduction(arg, '*.jpg')
arg1 = input("Please enter the Subject ID you would want to compare: ")
db = PostgresDB(password='1Idontunderstand', host='localhost',
                        database='postgres', user='postgres', port=5432)
conn = db.connect()
_, _, _, _ = dim.saveDim('l', 'svd', 'imagedata_l', 10, password ="1Idontunderstand", database="postgres")
# Change the database name in case you want to test with a different combination of features and dim
sub_matrix = dim.subMatrix(conn, 'imagedata_l_svd', arg1, mat=False)
for sub in sub_matrix:
    print('Subject: ', sub[0])
    print('Weight:', sub[1])
    print('\n')
>>>>>>> origin/Kovid_branch:BACKUP/task6.py
