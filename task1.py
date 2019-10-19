from dimReduction import dimReduction
import os
spath = input("Please enter the home directory for the images "
            "(Default: C:\\ASU\\Fall 2019\\MWDB\\Project\\Phase 2\\Dataset2\\) : ")
if spath == '':
    spath = 'C:\\ASU\\Fall 2019\\MWDB\\Project\\Phase 2\\Dataset2\\'
dim = dimReduction(spath, '*.jpg')
feature = input('Please choose a feature model - SIFT(s), Moments(m), LBP(l), Histogram(h): ')
if feature not in ('s', 'm', 'l', 'h'):
    print('Please enter a valid feature model!')
    exit()
technique = input('Please choose a dimensionality reduction technique - PCA(pca), SVD(svd), NMF(nmf), LDA(lda): ')
k = input('Please provide the number of latent semantics(k): ')
db = 'imagedata_' + feature
imgs_sort, feature_sort = dim.saveDim(feature, technique, db, int(k), password="password", database="mwdb")
path = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + 'Outputs'  +os.sep)
print(path)
print('\n')
print('Data Latent Semantics Saved to Output Folder!')
if not os.path.exists(path):
    os.makedirs(path)
dim.writeFile(imgs_sort, path + os.sep + 'Task1_Data_ls_{x}_{y}_{z}.txt'.format(x=feature,y=technique,z=k))
# print(imgs_sort)
print('\n')
print('Feature Latent Semantics Saved to Output Folder!')
dim.writeFile(feature_sort, path + os.sep + 'Task1_Feature_ls_{x}_{y}_{z}.txt'.format(x=feature,y=technique,z=k))
# print(feature_sort)
dim.imgViz(imgs_sort, spath)
dim.imgViz_feature(feature_sort, spath)
