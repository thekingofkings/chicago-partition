#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 09:08:14 2018

@author: hxw186

MCMC procedure to find the best partition
"""

from tract import Tract
from community_area import CommunityArea
from regression import NB_regression_training, NB_regression_evaluation
import random
from math import exp
import numpy as np
from shapely.ops import cascaded_union
import matplotlib.pyplot as plt


def initialize():
    global M, T, featureName, targetName, CA_maxsize, mae1, cnt, iter_cnt, \
        mae_series, mae_index, var_series,pop_variance1
    print "# initialize"
    random.seed(0)
    Tract.createAllTracts()
    CommunityArea.createAllCAs(Tract.tracts)
    featureName = CommunityArea.featureNames
    targetName = 'total'
    M = 100
    T = 10
    CA_maxsize = 30
    # Plot original community population distribution
    CommunityArea.visualizePopDist(fname='orig-pop-distribution')
    print "# sampling"
    CA_maxsize = 30
    mae1, _, _,errors = NB_regression_training(CommunityArea.features, featureName, targetName)
    pop_variance1 = np.var(CommunityArea.population)
    cnt = 0
    iter_cnt = 0
    mae_series = [mae1]
    var_series = [pop_variance1]
    mae_index = [0]


def get_f(ae,T,penalty=None,log=True):
    """
    compute the "energy function F".

    :param ae: Error measurement
    :param T: Temperature parameter
    :param penalty: value to penalize constrain object
    :param log: (Bool) Return f on log scale
    :return: the 'energy' of a given state
    """
    if penalty is None:
        lmbda = 0
    else:
        lmbda = penalty

    if log:
        return -(ae + lmbda) / T
    else:
        return np.exp(-(ae + lmbda) / T)

def get_gamma(f1,f2,log=True):
    """
    Compute gamma to be used in acceptance probability of proposed state
    :param f1: f ("energy function") of current state
    :param f2: f ("energy function") of proposed state
    :param log: (bool) Are f1 and f2 on log scale?
    :return: value for gamma, or probability of accepting proposed state
    """

    if log:
        alpha = f2 - f1
        return np.min((0,alpha))
    else:
        alpha = f2/f1
        return np.min((1,alpha))


def softmax(x,log=False):
    """
    Compute softmax of a vector x. Always subtract out max(x) to make more numerically stable.
    :param x: (array-like) vector x
    :param log: (bool) return log of softmax function
    :return: Vector of probabilities using softmax function
    """

    # Numerically stable softmax: softmax_stable = softmax(x - max_i(x))
    max_x =  np.max(x)
    x_centered = x - max_x
    if log:
        log_sum = np.log(np.sum(np.exp(x_centered)))
        return x_centered - log_sum
    else:
        exp_X = np.exp(x_centered)
        return exp_X / np.sum(exp_X)



def plotMcmcDiagnostics(mae_index,error_array,variance_array,fname='mcmc-diagnostics'):
    #x = range(len(error_array))
    # Two subplots, the axes array is 1-d
    f, axarr = plt.subplots(2, sharex=True,figsize=(12,8))
    axarr[0].plot(mae_index, np.array(error_array))
    axarr[0].set_title('Mean Absolute Error')
    axarr[1].plot(mae_index, np.array(variance_array))
    axarr[1].set_title('Population Variance (over communities)')

    plt.savefig(fname)
    plt.close()
    plt.clf()



def mcmcSamplerUniform(sample_func, update_sample_weight_func):
    """
    MCMC search for optimal solution.
    Input:
        sample_func is the sample proposal method.
        update_sample_weight_func updates sampling auxilary variables.
    Output:
        Optimal partition plot and training error decreasing trend.
    """
    global mae1, cnt, iter_cnt, pop_variance1
    print "# sampling"
    while cnt <= M:
        cnt += 1
        iter_cnt += 1
        # sample a boundary tract
        t = sample_func(Tract.boundarySet, 1)[0]
        t_flip_candidate = set()
        for n in t.neighbors:
            if n.CA != t.CA and n.CA not in t_flip_candidate:
                t_flip_candidate.add(n.CA)
        # sample a CA assignment to flip
        new_caid = t_flip_candidate.pop() if len(t_flip_candidate) == 1 else random.sample(t_flip_candidate, 1)[0]
        prv_caid = t.CA
        # check wether spatial continuity is guaranteed, if t is flipped
        ca_tocheck = CommunityArea.CAs[prv_caid].tracts
        del ca_tocheck[t.id]
        resulted_shape = cascaded_union([e.polygon for e in ca_tocheck.values()])
        ca_tocheck[t.id] = t
        if resulted_shape.geom_type == 'MultiPolygon':
            continue
        # CA size constraint
        if len(CommunityArea.CAs[new_caid].tracts) > CA_maxsize  \
            or len(CommunityArea.CAs[prv_caid].tracts) <= 1:
            continue
        
        # update communities features for evaluation
        t.CA = new_caid
        CommunityArea.updateCAFeatures(t, prv_caid, new_caid)
        # Get updated variance of population distribution
        pop_variance2 = np.var(CommunityArea.population)
        # evaluate new partition
        mae2, _, _ = NB_regression_training(CommunityArea.features, featureName, targetName)
        # Calculate acceptance probability --> Put on log scale
        # calculate f ('energy') of current and proposed states
        F1 = get_f(ae = mae1, T=T,penalty=pop_variance1,log=True)
        F2 = get_f(ae = mae2, T=T,penalty=pop_variance2,log=True)
        # Compute gamma for acceptance probability
        gamma = get_gamma(f1=F1,f2=F2,log=True)
        # Generate random number on log scale
        sr = np.log(random.random())
        update_sample_weight_func(mae1, mae2, t)

        if sr < gamma: # made progress
            mae_series.append(mae2)
            var_series.append(pop_variance2)
            print "Iteration {}: {} --> {} in {} steps".format(iter_cnt,mae1, mae2, cnt)
            # Update error, variance
            mae1, pop_variance1 = mae2, pop_variance2
            mae_index.append(iter_cnt)

            # update tract boundary set for next round sampling 
            Tract.updateBoundarySet(t)
            cnt = 0 # reset counter
            
            if len(mae_series) > 75 and np.std(mae_series[-50:]) < 3:
                # when mae converges
                print "converge in {} samples with {} acceptances \
                    sample conversion rate {}".format(iter_cnt, len(mae_series),
                                                len(mae_series) / float(iter_cnt))
                CommunityArea.visualizeCAs(fname="CAs-iter-final.png")
                CommunityArea.visualizePopDist(fname='final-pop-distribution')


                break

            if iter_cnt % 500 == 0:
                CommunityArea.visualizeCAs(fname="CAs-iter-{}.png".format(iter_cnt))
                CommunityArea.visualizePopDist(fname='pop-distribution-iter-{}'.format(iter_cnt))

        else:
            # restore communities features
            t.CA = prv_caid
            CommunityArea.updateCAFeatures(t, new_caid, prv_caid)


def mcmcSamplerSoftmax():
    """
    MCMC search for optimal solution.
    Input:
        sample_func is the sample proposal method.
        update_sample_weight_func updates sampling auxilary variables.
    Output:
        Optimal partition plot and training error decreasing trend.
    """
    global mae1, cnt, iter_cnt, pop_variance1
    print "# sampling"
    while cnt <= M:
        cnt += 1
        iter_cnt += 1

        # Learn regression to obtain community-level errors (current state)
        mae1, _, _,errors1 = NB_regression_training(CommunityArea.features, featureName, targetName)

        ca_probs = softmax(errors1,log=False)

        # Sample community -- probabilities derived from softmax of regression errors
        all_communities = CommunityArea.CAs
        sample_ca_id = np.random.choice(a=all_communities.keys(),size=1,replace=False,p=ca_probs)[0]
        sample_ca = all_communities[sample_ca_id]


        # Collect tracts within sampled community area that lie on community boundary
        sample_ca_boundary_tracts = []
        for tract in Tract.boundarySet:
            if tract.CA == sample_ca_id:
                sample_ca_boundary_tracts.append(tract)

        # Sample tract (on boundary) within previously sampled community area
        t = np.random.choice(a=sample_ca_boundary_tracts,size=1,replace=False)[0]

        # Find neighbors that lie in different community
        t_flip_candidate = set()
        for n in t.neighbors:
            if n.CA != t.CA and n.CA not in t_flip_candidate:
                t_flip_candidate.add(n.CA)
        # sample a CA assignment to flip
        new_caid = t_flip_candidate.pop() if len(t_flip_candidate) == 1 else random.sample(t_flip_candidate, 1)[0]
        prv_caid = t.CA
        # check wether spatial continuity is guaranteed, if t is flipped
        ca_tocheck = CommunityArea.CAs[prv_caid].tracts
        del ca_tocheck[t.id]
        resulted_shape = cascaded_union([e.polygon for e in ca_tocheck.values()])
        ca_tocheck[t.id] = t
        if resulted_shape.geom_type == 'MultiPolygon':
            continue
        # CA size constraint
        if len(CommunityArea.CAs[new_caid].tracts) > CA_maxsize \
                or len(CommunityArea.CAs[prv_caid].tracts) <= 1:
            continue

        # update communities features for evaluation
        t.CA = new_caid
        CommunityArea.updateCAFeatures(t, prv_caid, new_caid)
        # Get updated variance of population distribution
        pop_variance2 = np.var(CommunityArea.population)
        # evaluate new partition
        mae2, _, _,errors2 = NB_regression_training(CommunityArea.features, featureName, targetName)
        # Calculate acceptance probability --> Put on log scale
        # calculate f ('energy') of current and proposed states
        F1 = get_f(ae=mae1, T=T, penalty=pop_variance1, log=True)
        F2 = get_f(ae=mae2, T=T, penalty=pop_variance2, log=True)
        # Compute gamma for acceptance probability
        gamma = get_gamma(f1=F1, f2=F2, log=True)
        # Generate random number on log scale
        sr = np.log(random.random())


        if sr < gamma:  # made progress
            mae_series.append(mae2)
            var_series.append(pop_variance2)
            print "Iteration {}: {} --> {} in {} steps".format(iter_cnt, mae1, mae2, cnt)
            # Update error, variance
            mae1, pop_variance1 = mae2, pop_variance2
            mae_index.append(iter_cnt)

            # update tract boundary set for next round sampling
            Tract.updateBoundarySet(t)
            cnt = 0  # reset counter

            if len(mae_series) > 75 and np.std(mae_series[-50:]) < 3:
                # when mae converges
                print "converge in {} samples with {} acceptances \
                    sample conversion rate {}".format(iter_cnt, len(mae_series),
                                                      len(mae_series) / float(iter_cnt))
                CommunityArea.visualizeCAs(fname="CAs-iter-final.png")
                CommunityArea.visualizePopDist(fname='final-pop-distribution')

                break

            if iter_cnt % 500 == 0:
                CommunityArea.visualizeCAs(fname="CAs-iter-{}.png".format(iter_cnt))
                CommunityArea.visualizePopDist(fname='pop-distribution-iter-{}'.format(iter_cnt))

        else:
            # restore communities features
            t.CA = prv_caid
            CommunityArea.updateCAFeatures(t, new_caid, prv_caid)


def leaveOneOut_evaluation(year, info_str="optimal boundary"):
    """
    Leave-one-out evaluation the current partitino with next year crime rate.
    """
    CommunityArea._initializeCAfeatures(crimeYear=year)
    featureName = CommunityArea.featureNames
    targetName = 'total'
    print "leave one out with {} in {}".format(info_str, year)
    print NB_regression_evaluation(CommunityArea.features, featureName, targetName)
    
    

def naive_MCMC():
    initialize()
    # loo evaluation test data on original boundary
    leaveOneOut_evaluation(2011, "Administrative boundary")
    # restore training data
    CommunityArea._initializeCAfeatures(2010)

    mcmcSamplerUniform(random.sample, lambda ae1, ae2, t : 1)
    plotMcmcDiagnostics(mae_index=mae_index,error_array=mae_series,variance_array=var_series)
    leaveOneOut_evaluation(2011)




def adaptive_MCMC():
    initialize()
    # initialize adapative sampling variable
    ntrct = len(Tract.tracts)
    tractWeights = dict(zip(Tract.tracts.keys(), [1.0]*ntrct))
    
    def adaptive_sample(tractSet, k):
        tractIDs = [t.id for t in tractSet]
        sampleWeights = [tractWeights[tid] for tid in tractIDs]
        tmp = random.uniform(0, sum(sampleWeights))
        
        for tid in tractIDs:
            if tmp < tractWeights[tid]:
                return [Tract.tracts[tid]]
            else:
                tmp -= tractWeights[tid]
        return [None]
    
    
    def update_tractWeight(ae1, ae2, t):
        if ae1 < ae2:
            tractWeights[t.id] *= 0.8
        else:
            tractWeights[t.id] *= 1/0.8

    mcmcSamplerUniform(adaptive_sample, update_tractWeight)
    plotMcmcDiagnostics(mae_index=mae_index,error_array=mae_series,variance_array=var_series)


def MCMC_softmax_proposal():
    initialize()
    # loo evaluation test data on original boundary
    leaveOneOut_evaluation(2011, "Administrative boundary")
    # restore training data
    CommunityArea._initializeCAfeatures(2010)

    mcmcSamplerSoftmax(random.sample, lambda ae1, ae2, t: 1)
    plotMcmcDiagnostics(mae_index=mae_index, error_array=mae_series, variance_array=var_series)
    leaveOneOut_evaluation(2011)



if __name__ == '__main__':
    MCMC_softmax_proposal()


