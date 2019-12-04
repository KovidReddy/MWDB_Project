from classify import classify

label = input('Please enter the Image you want to compare: ')
n = input('Please enter the number of neighbors: ')
k = input('Please enter the number of hashes (k): ')
l = input('Please enter the number of layers (l): ')

cls = classify()
cls.relevanceFeedback(n=int(n), label=label, k=int(k), l=int(l))
