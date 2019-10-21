from dimReduction import dimReduction
import os
path = input("Please enter the home directory for the images "
            "(Default: C:\\Users\\pylak\\Documents\\Fall 2019\\MWDB\\Project\\Phase1\\Hands_test2\\) : ")
if path == '':
    path = 'C:\\Users\\pylak\\Documents\\Fall 2019\\MWDB\\Project\\Phase1\\Dataset2\\'
dim = dimReduction(path, '*.jpg')
for feature in ['m', 'l', 'h', 's']:
    for technique in ['pca', 'nmf', 'lda', 'svd']:
        db = 'imagedata_' + feature
        imgs_sort, feature_sort = dim.saveDim(feature, technique, db, 10, password="abcdefgh", database="mwdb")
        path = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + 'Outputs'  +os.sep)
        # print(path)
        # print('\n')
        # print('Data Latent Semantics Saved to Output Folder!')
        dim.writeFile(imgs_sort, path + os.sep + 'Task1_Data_ls_{x}_{y}_{z}.txt'.format(x=feature,y=technique,z=10))
        # print('\n')
        # print('Feature Latent Semantics Saved to Output Folder!')
        dim.writeFile(feature_sort, path + os.sep + 'Task1_Feature_ls_{x}_{y}_{z}.txt'.format(x=feature,y=technique,z=10))