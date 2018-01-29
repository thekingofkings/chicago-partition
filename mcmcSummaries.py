import pandas as pd
from sklearn.metrics import adjusted_rand_score
from itertools import combinations, product
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series


def plotMcmcDiagnostics(iter_cnt,mae_index,error_array,f_array,std_array,lmbda=0.75,fname='mcmc-diagnostics'):
    #x = range(len(error_array))
    # Two subplots, the axes array is 1-d

    if iter_cnt is None:
        iter_cnt = "completed"

    f, axarr = plt.subplots(3, sharex=True,figsize=(12,8))
    axarr[0].plot(mae_index, np.array(error_array))
    axarr[0].set_title('Mean Absolute Error -- Iterations: {}'.format(iter_cnt))
    axarr[1].plot(mae_index, np.array(std_array))
    axarr[1].set_title('Standard Deviation of Community Size (in pop)')
    axarr[2].plot(mae_index, f_array)
    axarr[2].set_title('f - lambda = {}'.format(lmbda))

    plt.savefig("plots/" + fname)
    plt.close()
    plt.clf()


def writeSimulationOutput(project_name,error,n_iter_conv,accept_rate):

    fname = "output/{}-final-output.txt".format(project_name)
    f = open(fname,'w')
    f.write("error: {:.4f}\n".format(error))
    f.write("iterations: {}\n".format(n_iter_conv))
    f.write("acceptance rate: {:.4f}\n".format(accept_rate))
    f.close()




def getPartitionFromFile(fname):
    return pd.read_csv("output/" + fname,header=None,index_col=None).values.flatten()

def getSimulationOutput(fname):

    d = {}
    with open(fname) as f:
        for line in f:
            line_split = line.split(":")
            d[line_split[0]] = float(line_split[1].strip())

    return d

def getSimulationSummaryStats(project_name,n_sim):

    sims = range(0,n_sim)
    versions = ['v' + str(x+1) for x in sims]

    output_dicts = []
    for v in versions:
        fname = "output/{}-{}-final-output.txt".format(project_name,v)
        output_dicts.append(getSimulationOutput(fname))

    agg_dict = {}
    for key in output_dicts[0].keys():
        agg_dict[key] = [d[key] for d in output_dicts]


    mean_dict = {}
    for key in agg_dict.keys():
        mean = np.mean(agg_dict[key])
        sd = np.std(agg_dict[key])
        print "{:s} - Mean {:s} {:.2f} ({:.2f})".format(project_name,key,mean,sd)
        mean_dict[key] = mean
    return mean_dict


def computeRandScore(partition_1, partition_2):
    rand_score = adjusted_rand_score(partition_1,partition_2)

    return rand_score


def getCombinations(x1,x2=None):

    if x2 is None:
        c = combinations(x1,2)
    else:
        c = product(x1,x2)

    return c

def randIdxSimulation(project_name1, project_name2=None,n_sim=10):
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
    print "{:s} - Mean Adjusted Rand Index {:.4f} ({:.2f})".format(project_name1,mean_rand,sd_rand)
    return mean_rand, sd_rand



if __name__ == '__main__':
    print "Rand Index:"
    print "-----------"
    randIdxSimulation('softmax-sampler',n_sim=10)
    randIdxSimulation('q-learning', n_sim=10)
    randIdxSimulation('naive-sampler',n_sim=10)
    print "------------"
    getSimulationSummaryStats('naive-sampler',n_sim=10)
    getSimulationSummaryStats('softmax-sampler', n_sim=10)
    getSimulationSummaryStats('q-learning', n_sim=10)




