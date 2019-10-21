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

dim = dimReduction(path, '*.jpg')

for feature in ['m', 'l', 'h','s']:
    for technique in ['pca', 'nmf', 'lda', 'svd']:
        db = 'imagedata_' + feature
        imgs_sort, feature_sort, data_latent, feature_latent = dim.saveDim(feature, technique, db, 10, password="abcdefgh", database="mwdb", label = 'right', meta = False)

        path = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + 'Outputs' +os.sep)
        # print(path)
        # print('\n')
        # print('Data Latent Semantics Saved to Output Folder!')
        dim.writeFile(imgs_sort, path + os.sep + 'Task3_Data_ls_{x}_{y}_{z}.txt'.format(x=feature,y=technique,z=10))
        # print('\n')
        # print('Feature Latent Semantics Saved to Output Folder!')
        dim.writeFile(feature_sort, path + os.sep + 'Task3_Feature_ls_{x}_{y}_{z}.txt'.format(x=feature,y=technique,z=10))