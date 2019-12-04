"""
Still working on this
"""

from imageProcess import imageProcess
import pickle
import math
import numpy as np
from classify import classify
import pandas as pd

imgP = imageProcess()
meta_data = imgP.readMetaData()
personlization = {}
c = classify()
res = c.fetchData()
# for m in meta_data:
#     personlization[m[1]] = 0
#     print("calculating")
# with open('personalization.pickle', 'wb') as handle:
#     print("saving personalization")
#     pickle.dump(personlization, handle)
print("done saving personalization")
def similarity(res, id):
    s = {}
    temp = res[id]
    for i in res:
        dist = 1/(1+(math.sqrt(np.sum([((a-b) ** 2) for (a, b) in zip(temp, res[i])]))))
        s[i] = dist
    s = sorted(s.items(), key = lambda x : x[1])
    return s
data = []
for i in res:
    weights = similarity(res, i)
    temp_dict = {}
    for w in weights:
        temp_dict[w[0]] = w[1]
    data.append(temp_dict)
    print("done for: ", i)
df = pd.DataFrame(data, index=res.keys())
with open('similarity.pickle', 'wb') as handle:
    print("saving similarity")
    pickle.dump(df, handle)
