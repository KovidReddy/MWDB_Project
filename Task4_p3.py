from classify import classify

# task = int(input('Please enter the task you want: '))
feature = input('Please choose a feature model - SIFT(s), Moments(m), LBP(l), Histogram(h): ')
if feature not in ('s', 'm', 'l', 'h'):
    print('Please enter a valid feature model!')
    exit()
technique = input('Please choose a dimensionality reduction technique - PCA(pca), SVD(svd), NMF(nmf), LDA(lda): ')
algo = input('Please choose the algorithm for the feedback mechanism: '
                     '(SVM -svm, Decision Tree -dt, PPR -ppr):')
cls = classify(feature = feature, dim = technique)

if algo == 'svm':
    cls.svmClassify()
elif algo == 'dt':
    cls.decisionTreeClassify()
elif algo == 'ppr':
    cls.PPR()
# cls = classify()
# cls.relevanceFeedback(n=int(n), label=label, k=int(k), l=int(l))
