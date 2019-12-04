from classify import classify
technique = input('Please choose a feature model: (lbp - l, hog - h, sift - s, moments - m ) ')
algo = input('Please choose dim reduction technique: pca, svd, nmf, lda: ')
cls = classify()
cls.clusterClassify()