import pandas as pd
from sklearn.metrics import adjusted_rand_score
from itertools import combinations, product
import numpy as np

def getPartitionFromFile(fname):
    return pd.read_csv("output/" + fname,header=None,index_col=None).values.flatten()

def computeRandScore(partition_1, partition_2):
    rand_score = adjusted_rand_score(partition_1,partition_2)

    return rand_score


def getCombinations(x1,x2=None):

    if x2 is None:
        c = combinations(x1,2)
    else:
        c = product(x1,x2)

    return c

def randIdxSimulation(project_name1, project_name2=None):

    n_sim = 4
    sims = range(0,n_sim)
    versions = ['v' + str(x+1) for x in sims]
    partitions = []

    if project_name2 is None:
        for v in versions:
            fname = '{}-{}-final-partition.txt'.format(project_name1,v)
            partitions.append(getPartitionFromFile(fname))
        combos = getCombinations(sims)

        rand_scores = []
        for pair in combos:
            p1,p2 = partitions[pair[0]],partitions[pair[1]]
            adj_rand = computeRandScore(p1,p2)
            rand_scores.append(adj_rand)
    else:
        partitions2 = []
        for v in versions:
            fname1 = '{}-{}-final-partition.txt'.format(project_name1,v)
            partitions.append(getPartitionFromFile(fname1))
            fname2 = '{}-{}-final-partition.txt'.format(project_name2, v)
            partitions2.append(getPartitionFromFile(fname2))
        combos = getCombinations(sims,sims)
        rand_scores = []
        for pair in combos:
            p1, p2 = partitions[pair[0]], partitions2[pair[1]]
            adj_rand = computeRandScore(p1, p2)
            rand_scores.append(adj_rand)

        project_name1 = project_name1 + " & " + project_name2

    mean_rand = np.round(np.mean(rand_scores),4)
    sd_rand = np.round(np.std(rand_scores),4)
    print "{} - Mean Adjusted Rand Index {} ({})".format(project_name1,mean_rand,sd_rand)
    return mean_rand, sd_rand



if __name__ == '__main__':
    randIdxSimulation('softmax-sampler')
    randIdxSimulation('naive-sampler')
    randIdxSimulation('naive-sampler','softmax-sampler')




