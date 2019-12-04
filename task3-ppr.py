from matplotlib import pyplot as plt
import numpy as np
import math
from classify import classify
import pandas as pd
import matplotlib.image as pltimg
import matplotlib
from imageProcess import imageProcess

c = classify()
imgP = imageProcess()
res = c.fetchData(table_name="ppr_imagedata_h")

def similarity(res, id, k):
    s = {}
    temp = res[id]
    for i in res:
        dist = 1/(1+(math.sqrt(np.sum([((a-b) ** 2) for (a, b) in zip(temp, res[i])]))))#math.sqrt(np.sum([((a-b) ** 2) for (a, b) in zip(temp, res[i])]))
        s[i] = dist
    s = sorted(s.items(), key = lambda x : x[1], reverse=True)
    return s[0:k+1]

k = int(input("Enter k (Number of edges from each vertex):"))
K = int(input("Enter K (Number of images to be returned after personalization): "))
id1 = input("Enter first id: (Hand_0008333)")
id2 = input("Enter Second id: (Hand_0006183)")
id3 = input("Enter third id: (Hand_0000074)")
ids = [id1, id2, id3]
personalization = {}
for i in res:
    if i in ids:
        personalization[i] = 1/len(ids)
    else:
        personalization[i] = 0
data = []
for i in res:
    weights = similarity(res, i, k)
    temp_dict = {}
    for w in weights:
        temp_dict[w[0]] = w[1]
    for key in res.keys():
        if key not in temp_dict:
            temp_dict[key] = 0
    data.append(temp_dict)
df = pd.DataFrame(data, index=res.keys())
M = df / np.sum(df, axis=0)

teleportation_matrix = np.zeros(len(df))
pageRankScores = np.zeros(len(df))

for node_id in ids:
    teleportation_matrix[list(res.keys()).index(node_id)] = 1 / len(ids)
    pageRankScores[list(res.keys()).index(node_id)] = 1 / len(ids)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(M)

i = 0

while True:
    oldPageRankScores = pageRankScores
    pageRankScores = (0.85 * np.dot(M, pageRankScores)) + (0.15 * teleportation_matrix)
    if np.linalg.norm(pageRankScores - oldPageRankScores) < 0.0000001:
        break
    i += 1

pageRankScores = pageRankScores / sum(pageRankScores)
print("printing pagerank scores")
z = 0
res = {}
for i in personalization:
    res[i] = pageRankScores[z]
    z += 1
res = sorted(res.items(), key = lambda x : x[1])
print(res[0:K])
l = res[0:K]

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