from dimReduction import dimReduction
from PostgresDB import PostgresDB
import numpy as np
from imageProcess import imageProcess
    

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
image = input("Insert the name of your image: ")
k = input('Please provide the number of latent semantics(k): ')
label = input("Which label do you want: ")
label = label.replace(" ", "_")
db = 'imagedata_' + feature + '_' + technique + '_' + label
dim = dimReduction(path, '*.jpg')
task4 = imageProcess(path)
# filteredImage = task4.CSV(label)
task4.similarity(feature, technique, db, int(k), image, label)

# print('\n')
# print('Data Latent Semantics Saved to Output Folder!')
# dim.writeFile(imgs_sort, 'Output\\Task1\\Data_ls_{x}_{y}_{z}.txt'.format(x=feature,y=technique,z=k))
# print('\n')
# print('Feature Latent Semantics Saved to Output Folder!')
# dim.writeFile(imgs_sort, 'Output\\Task1\\Feature_ls_{x}_{y}_{z}.txt'.format(x=feature,y=technique,z=k))