import pprint

import networkx as nx
import numpy as np
import math
import pandas as pd
from matplotlib import pyplot as plt
from collections import Counter
from imageProcess import imageProcess
from classify import classify
import matplotlib.image as pltimg
import matplotlib
c = classify()
res = c.fetchData(table_name="ppr_imagedata_h")
imgP = imageProcess()
gr = nx.DiGraph()
gr.add_nodes_from(res.keys())

def similarity(res, id, k):
    s = {}
    temp = res[id]
    for i in res:
        dist = math.sqrt(np.sum([((a-b) ** 2) for (a, b) in zip(temp, res[i])]))
        s[i] = dist
    s = sorted(s.items(), key = lambda x : x[1])
    return s[0:k+1]

k = 4 #int(input("Enter k (Number of edges from each vertex):"))
K = 10 #int(input("Enter K (Number of images to be returned after personalization): "))
id1 = "Hand_0008333"#input("Enter first id: ")
id2 = "Hand_0006183"#input("Enter Second id: ")
id3 = "Hand_0000074"#input("Enter third id: ")
ids = [id1, id2, id3]
personalization = {}
for i in res:
    if i in ids:
        personalization[i] = 1
    else:
        personalization[i] = 0
data = []
for i in res:
    weights = similarity(res, i, k)

    for w in weights:
        gr.add_edge(i, w[0], weight=w[1], length=w[1])

#nx.draw_networkx(gr, pos=nx.circular_layout(gr), with_labels=True)

ppr = nx.pagerank(gr, alpha=0.85, personalization=personalization)
l = Counter(ppr).most_common(K)
pprint.pprint(l)

font = {'size': 8}
matplotlib.rc('font', **font)

w = 10
h = 15
fig = plt.figure(figsize=(10, 8))
columns = 5
rows = 3
j = 1
for i in ids:
    img = pltimg.imread(imgP.paths()[0] + i + ".jpg")
    ax1 = fig.add_subplot(rows, columns, j)
    ax1.title.set_text("test: " + i)
    plt.imshow(img)
    plt.axis('off')
    j+=1
j=0
for i in range(4, K + 4):
    img = pltimg.imread(imgP.paths()[0] + l[j][0] + ".jpg")
    ax1 = fig.add_subplot(rows, columns, i)
    ax1.title.set_text(l[j][0])
    plt.imshow(img)
    plt.axis('off')
    j += 1
plt.suptitle("Personalized PageRank")
plt.show()


