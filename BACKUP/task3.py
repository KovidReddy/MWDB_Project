from dimReduction import dimReduction
from PostgresDB import PostgresDB
import numpy as np
from imageProcess import imageProcess
import os



path = input("Please enter the home directory for the images "
            "(Default: C:\\Users\\pylak\\Documents\\Fall 2019\\MWDB\\Project\\Phase1\\Dataset\\) : ")
if path == '':
    path = 'C:\\Users\\pylak\\Documents\\Fall 2019\\MWDB\\Project\\Phase1\\Dataset\\'
dim = dimReduction(path, '*.jpg')
feature = input('Please choose a feature model - SIFT(s), Moments(m), LBP(l), Histogram(h): ')
if feature not in ('s', 'm', 'l', 'h'):
    print('Please enter a valid feature model!')
    exit()
technique = input('Please choose a dimensionality reduction technique - PCA(pca), SVD(svd), NMF(nmf), LDA(lda): ')
k = input('Please provide the number of latent semantics(k): ')
label = input("Which label do you want: ")
db = 'imagedata_' + feature
dim = dimReduction(path, '*.jpg')
# task3 = imageProcess("/home/anhnguyen/ASU/CSE-515/Project/Phase 2/Project - Phase 2/Data/testset1/")
# filterImage = task3.CSV(label)
imgs_sort, feature_sort, data_latent, feature_latent = dim.saveDim(feature, technique, db, int(k), password = "1Idontunderstand", database = "postgres", label = label, meta = True)

path = os.path.normpath(os.getcwd()  + os.sep + os.pardir + os.sep + 'Phase1\\Outputs'  +os.sep)
# print(path)
print('\n')
print('Data Latent Semantics Saved to Output Folder!')
# print(data_latent)
dim.writeFile(data_latent, path + os.sep + 'Task3_Data_ls_{x}_{y}_{z}_{l}.txt'.format(x=feature,y=technique,z=k, l = label))
print('\n')
print('Feature Latent Semantics Saved to Output Folder!')
# print(feature_latent)
dim.writeFile(feature_latent, path + os.sep + 'Task3_Feature_ls_{x}_{y}_{z}.txt'.format(x=feature,y=technique,z=k, l = label))
# print(feature_sort)
dim.imgViz(imgs_sort)
dim.imgViz_feature(feature_sort)