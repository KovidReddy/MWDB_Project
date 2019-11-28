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
# Change the database name in case you want to test with a different combination of features and dim
_, _, _, _ = dim.saveDim('l', 'svd', 'imagedata_l', 10, password ="1Idontunderstand", database="postgres")
sub_matrix = dim.subMatrix(conn, 'imagedata_l_svd', mat=True)
dim.writeFile(sub_matrix, 'Outputs\\task7.txt')

for idx, sub in enumerate(sub_matrix):
    print('\nLatent Semantic {x}'.format(x=idx+1))
    for s in sub:
        print('Subject: ', s[0])
        print('Weight: ', s[1])
# dim.imgViz(sub_matrix)
