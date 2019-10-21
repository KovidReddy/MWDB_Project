from dimReduction import dimReduction
import os
path = input("Please enter the home directory for the images "
            "(Default: C:\\Users\\pylak\\Documents\\Fall 2019\\MWDB\\Project\\Phase1\\Hands_test2\\) : ")
if path == '':
    path = 'C:\\Users\\pylak\\Documents\\Fall 2019\\MWDB\\Project\\Phase1\\Hands_test2\\'
dim = dimReduction(path, '*.jpg')
feature = input('Please choose a feature model - SIFT(s), Moments(m), LBP(l), Histogram(h): ')
if feature not in ('s', 'm', 'l', 'h'):
    print('Please enter a valid feature model!')
    exit()
technique = input('Please choose a dimensionality reduction technique - PCA(pca), SVD(svd), NMF(nmf), LDA(lda): ')
k = input('Please provide the number of latent semantics(k): ')
db = 'imagedata_' + feature
imgs_sort, feature_sort, data_latent, feature_latent = dim.saveDim(feature, technique, db, int(k), password = "abcdefgh", database = "mwdb")
path = os.path.normpath(os.getcwd()  + os.sep + os.pardir + os.sep + 'Outputs')
print(path)
print('\n')
print('Data Latent Semantics Saved to Output Folder!')
print(data_latent)
dim.writeFile(data_latent, path + os.sep + 'Task1_Data_ls_{x}_{y}_{z}.txt'.format(x=feature,y=technique,z=k))
print('\n')
print('Feature Latent Semantics Saved to Output Folder!')
print(feature_latent)
dim.writeFile(feature_latent, path + os.sep + 'Task1_Feature_ls_{x}_{y}_{z}.txt'.format(x=feature,y=technique,z=k))
print(feature_sort)
dim.imgViz(imgs_sort)
dim.imgViz_feature(feature_sort)

