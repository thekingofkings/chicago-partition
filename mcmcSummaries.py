import pandas as pd
from sklearn.metrics import adjusted_rand_score

def getPartitionFromFile(fname):
    return pd.read_csv(fname,header=None,index_col=None).values.flatten()

def computeRandScore(partition_1, partition_2):
    rand_score = adjusted_rand_score(partition_1,partition_2)
    print "Adjusted Rand Index {0:.4f}".format(rand_score)
    return rand_score




if __name__ == '__main__':
    p1 = getPartitionFromFile('output/softmax-sampler-final-partition-1.txt')
    p3 = getPartitionFromFile('output/softmax-sampler-final-partition-2.txt')
    computeRandScore(p1,p3)


