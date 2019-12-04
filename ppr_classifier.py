import networkx as nx
from classify import classify
import numpy as np
import math
import glob
from imageProcess import imageProcess
import pprint
from collections import Counter

c = classify()
img = imageProcess()
res = c.fetchData()
#29
labels=["d", "d", "d", "d", "d", "d", "d", "d", "d", "d", "d", "d", "d", "d", "p", "p", "p", "p", "p", "p", "p", "p", "p", "p", "p", "p", "p", "p", "p"]
gr = nx.DiGraph()
gr.add_nodes_from(res.keys())
threshold = 6
folder = "C:\\Users\\nemad\\PycharmProjects\\new1\\venv\\rinku\\"#input("Enter folder path:")
def similarity(res, id, k):
    s = {}
    temp = res[id]
    for i in res:
        dist = math.sqrt(np.sum([((a-b) ** 2) for (a, b) in zip(temp, res[i])]))
        s[i] = dist
    s = sorted(s.items(), key = lambda x : x[1])
    return s[0:k]
for i in res:
    weights = similarity(res, i, threshold)
    for w in weights:
        gr.add_edge(i, w[0], weight=w[1], length=w[1])

personalization = {}
index = 0
for i in res:
    if labels[index] == "d":
        personalization[i] = 1
    else:
        personalization[i] = 0
    index += 1

K=5
for filename in glob.glob(folder + "*.jpg"):
    h_val = img.hog_process(filename)
    s = {}
    personalization[filename] = 1
    for i in res:
        dist = math.sqrt(np.sum([((a - b) ** 2) for (a, b) in zip(h_val, res[i])]))
        s[i] = dist
    s = sorted(s.items(), key=lambda x: x[1])
    gr.add_node(filename)
    for w in s[0:threshold]:
        gr.add_edge(i, w[0], weight=w[1], length=w[1])
    ppr = nx.pagerank(gr, alpha=0.75, personalization=personalization)
    print(filename)
    pprint.pprint(Counter(ppr).most_common(K))
    gr.remove_node(filename)
    del personalization[filename]

personalization = {}
index = 0
for i in res:
    if labels[index] == "p":
        personalization[i] = 1
    else:
        personalization[i] = 0
    index += 1

K=5
for filename in glob.glob(folder + "*.jpg"):
    h_val = img.hog_process(filename)
    s = {}
    personalization[filename] = 1
    for i in res:
        dist = math.sqrt(np.sum([((a - b) ** 2) for (a, b) in zip(h_val, res[i])]))
        s[i] = dist
    s = sorted(s.items(), key=lambda x: x[1])
    gr.add_node(filename)
    for w in s[0:threshold]:
        gr.add_edge(i, w[0], weight=w[1], length=w[1])
    ppr = nx.pagerank(gr, alpha=0.75, personalization=personalization)
    print(filename)
    pprint.pprint(Counter(ppr).most_common(K))
    gr.remove_node(filename)
    del personalization[filename]