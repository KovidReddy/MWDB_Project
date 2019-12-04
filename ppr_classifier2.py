import networkx as nx
from classify import classify
import numpy as np
import pandas as pd
import math
import glob
from imageProcess import imageProcess
from matplotlib import pyplot as plt
import matplotlib.image as pltimg
import pprint
from collections import Counter
import csv
import ppr_helper
c = classify()
imgP = imageProcess()
res = c.fetchData(table_name="ppr_imagedata_h")

threshold = 6
folder = input("Enter test folder path: (C:\\Users\\nemad\\Downloads\\MWDB\\project\\phase3_sample_data\\phase3_sample_data\\Unlabelled\\Set_2\\)")
def similarity(res, id):
    s = {}
    temp = res[id]
    for i in res:
        dist = 1/(1+(math.sqrt(np.sum([((a-b) ** 2) for (a, b) in zip(temp, res[i])]))))
        s[i] = dist
    s = sorted(s.items(), key = lambda x : x[1])
    return s
meta_data_verify = imgP.readMetaData()
meta_file="C:\\Users\\nemad\\Downloads\\MWDB\\project\\phase3_sample_data\\phase3_sample_data\\labelled_set2.csv"
with open(meta_file, 'r') as file:
    csv_reader = csv.reader(file)
    meta_file = []
    for idx, row in enumerate(csv_reader):
        if idx == 0:
            continue
        sub_id = row[1]
        id = row[8].split('.')[0]
        gender = row[3]
        orientation = row[7].split(' ')
        accessories = row[5]
        meta_file.append([sub_id, id, gender, orientation[0], orientation[1], accessories])

meta_data = meta_file

dorsal_ids = []
palmar_ids = []
d_personalization = {}
p_personalization = {}
p = d = 0
for m in meta_data:
    if m[3] == 'dorsal':
        dorsal_ids.append(m[1])
        d+=1
    else:
        palmar_ids.append(m[1])
        p+=1
#print("TRAINING DATA HAS ", p, "PALMAR and ", d, "DORSAL IMAGES")

for i in res:
        d_personalization[i] = 0

data = []
for i in res:
    weights = similarity(res, i)
    temp_dict = {}
    for w in weights:
        temp_dict[w[0]] = w[1]
    data.append(temp_dict)
df = pd.DataFrame(data, index=res.keys())

for i in res:
    weights = similarity(res, i)
img_no = 1
w = 10
h = 15
fig = plt.figure(figsize=(10, 8))
columns = 10
rows = 10
final_Result = {}
for filename in glob.glob(folder + "*.jpg"):
    temp = df
    h_val = imgP.hog_process(filename)
    filename = filename[-16:]
    filename = filename[0:len(filename)-4]
    s = {}
    col = []
    for i in res:
        dist = math.sqrt(np.sum([((a - b) ** 2) for (a, b) in zip(h_val, res[i])]))
        s[i] = dist
        col.append(dist)
    df1 = pd.DataFrame([s], index=[filename])
    temp = pd.concat([temp, df1])
    col.append(0)
    temp[filename] = col
    d_personalization[filename] = 1
    pageRankScores = np.zeros(len(d_personalization))
    teleportation_matrix = np.zeros(len(d_personalization))
    t = 0
    for i in d_personalization:
        pageRankScores[t] = d_personalization[i]
        teleportation_matrix[t] = d_personalization[i]
        t += 1

    pageRankScores = ppr_helper.ppr(temp, pageRankScores, teleportation_matrix, 0.85)
    z = 0
    res1 = {}
    for i in d_personalization:
        if i != filename:
            res1[i] = pageRankScores[z]
        z += 1
    res1 = sorted(res1.items(), key=lambda x: x[1], reverse=True)
    del d_personalization[filename]
    prob_d = res1[0]
    d = 0
    p = 0
    for tp in range(5):
        cl = res1[tp][0]
        if cl in dorsal_ids:
            d += res1[tp][1]
        else:
            p += res1[tp][1]
    if d > p:
        print(filename, "is dorsal with probability", d)
        title = "dorsal"
    else:
        print(filename, "is palmar with probability", p)
        title = "palmar"
    final_Result[filename] = title
    img = pltimg.imread(folder + filename + ".jpg")
    ax1 = fig.add_subplot(rows, columns, img_no)
    ax1.title.set_text(title)
    plt.imshow(img)
    plt.axis('off')
    img_no += 1
verification_d = {}
for i in meta_data_verify:
    verification_d[i[1]] = i[3]

count = 0
for i in final_Result:
    if final_Result[i] == verification_d[i]:
        count += 1
print("accuracy ", count/len(final_Result)*100)
plt.suptitle("Personalized PageRank Classifier")
plt.show()

