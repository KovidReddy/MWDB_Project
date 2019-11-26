from dimReduction import dimReduction
from PostgresDB import PostgresDB
import os
path = input("Please enter the home directory for the images "
            "(Default: C:\\Users\\pylak\\Documents\\Fall 2019\\MWDB\\Project\\Phase1\\Dataset\\) : ")
if path == '':
    path = 'C:\\Users\\pylak\\Documents\\Fall 2019\\MWDB\\Project\\Phase1\\Dataset\\'
dim = dimReduction(path, '*.jpg')
feature = input('Please choose a feature model - SIFT(s), Moments(m), LBP(l), Histogram(h): ')
image = input("Insert the name of your image: ")
if feature not in ('s', 'm', 'l', 'h'):
    print('Please enter a valid feature model!')
    exit()
technique = input('Please choose a dimensionality reduction technique - PCA(pca), SVD(svd), NMF(nmf), LDA(lda): ')
k = input('Please provide the number of latent semantics(k): ')
label = input('Please provide the label: ')
db = PostgresDB(password='mynhandepg', host='localhost',
                        database='mwdb', user='postgres', port=5432)
conn = db.connect()
x =dim.classifyImgDeTree(conn, feature, image, label, technique)
if x[0] == 1:
    print('Image {id} belongs to label {l}'.format(id=image, l=label))
else:
    print('Image {id} doesn\'t belongs to label {l}'.format(id=image, l=label))