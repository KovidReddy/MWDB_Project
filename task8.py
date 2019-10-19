from dimReduction import dimReduction
from PostgresDB import PostgresDB
arg = input("Please enter the home directory for the images "
            "(Default: C:\\Users\\pylak\\Documents\\Fall 2019\\MWDB\\Project\\Phase1\\Hands_test2\\) : ")
if arg == '':
    arg = 'C:\\Users\\pylak\\Documents\\Fall 2019\\MWDB\\Project\\Phase1\\Hands_test2\\'
dim = dimReduction(arg, '*.jpg')
db = PostgresDB(password='1Idontunderstand', host='localhost',
                        database='postgres', user='postgres', port=5432)
conn = db.connect()
bin_matrix, feature_matrix = dim.binMat(conn, 'imagedata_m_pca')
# code to print
print(bin_matrix[0][0])
print(feature_matrix)
