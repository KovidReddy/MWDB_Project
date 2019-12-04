import numpy as np
import math
"""
temp: similarity matrix
pageRankScores, teleportation_matrix: personalization matrix
"""
def ppr(temp, pageRankScores, teleportation_matrix, beta = 0.85):
    M = temp / np.sum(temp, axis=0)

    i = 0
    while True:
        oldPageRankScores = pageRankScores
        pageRankScores = (beta * np.dot(M, pageRankScores)) + ((1 - beta) * teleportation_matrix)
        if np.linalg.norm(pageRankScores - oldPageRankScores) < 0.0000001:
            break
        i += 1
    pageRankScores = pageRankScores / sum(pageRankScores)
    return pageRankScores
    
def similarity(res, id):
    s = {}
    temp = res[id]
    # print("check")
    # print(res, id)
    for i in res:
        dist = 1/(1+(math.sqrt(np.sum([((a-b) ** 2) for (a, b) in zip(temp, res[i])]))))
        s[i] = dist
    s = sorted(s.items(), key = lambda x : x[1])
    return s