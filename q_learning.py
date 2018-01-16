#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 07:57:36 2018

@author: hj

Q-learning as adaptive MCMC method

"""

from tract import Tract
from community_area import CommunityArea
from regression import NB_regression_training, NB_regression_evaluation
from shapely.ops import cascaded_union
import matplotlib.pyplot as plt
import random
import numpy as np
from MCMC import leaveOneOut_evaluation, get_f, get_gamma



def initialize():
    global featureName, targetName, M, T, CA_maxsize, mae1, mae_series, mae_index, \
        iter_cnt, pop_variance1, var_series
    print "# initialize"
    random.seed(0)
    Tract.createAllTracts()
    CommunityArea.createAllCAs(Tract.tracts)
    featureName = CommunityArea.featureNames
    targetName = 'total'
    M = 100
    T = 10
    CA_maxsize = 30
    mae1, _, _, errors = NB_regression_training(CommunityArea.features, featureName, targetName)
    pop_variance1 = np.var(CommunityArea.population)
    iter_cnt = 0
    mae_series = [mae1]
    var_series = [pop_variance1]
    mae_index = [0]



def sample_once():
    """
    Sample one tract, and return the tract if it is valid.
    Valid condition:
        The spatial continuity is guaranteed.
        The size constrait of each CA is guaranteed.
    Output:
        The tract and two effected CAs, if the sample is valid.
        Return None, otherwise.
    """
    # sample a boundary tract
    t = random.sample(Tract.boundarySet, 1)[0]
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
        return None
    # CA size constraint
    if len(CommunityArea.CAs[new_caid].tracts) > CA_maxsize  \
        or len(CommunityArea.CAs[prv_caid].tracts) <= 1:
        return None
    t.CA = new_caid
    return (t, prv_caid, new_caid)



if __name__ == '__main__':
    initialize()
    
    # loo evaluation test data on original boundary
    leaveOneOut_evaluation(2011, "Administrative boundary")
    # restore training data
    CommunityArea._initializeCAfeatures(2010)
    
    print "# sampling"
    while True:
        iter_cnt += 1
        
        sample_res = sample_once()
        if sample_res == None:
            continue
        t, prv_caid, new_caid = sample_res
        # update communities features for evaluation
        CommunityArea.updateCAFeatures(*sample_res)
        # Get updated variance of population distribution
        pop_variance2 = np.var(CommunityArea.population)
        # evaluate new partition
        mae2, _, _, _ = NB_regression_training(CommunityArea.features, featureName, targetName)
        # Calculate acceptance probability --> Put on log scale
        # calculate f ('energy') of current and proposed states
        F1 = get_f(ae = mae1, T=T,penalty=pop_variance1,log=True)
        F2 = get_f(ae = mae2, T=T,penalty=pop_variance2,log=True)
        # Compute gamma for acceptance probability
        gamma = get_gamma(f1=F1,f2=F2,log=True)
        # Generate random number on log scale
        sr = np.log(random.random())
#        update_sample_weight_func(mae1, mae2, t)

        if sr < gamma: # made progress
            mae_series.append(mae2)
            var_series.append(pop_variance2)
            print "Iteration {}: {} --> {}".format(iter_cnt, mae1, mae2)
            # Update error, variance
            mae1, pop_variance1 = mae2, pop_variance2
            mae_index.append(iter_cnt)

            # update tract boundary set for next round sampling 
            Tract.updateBoundarySet(t)
            
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