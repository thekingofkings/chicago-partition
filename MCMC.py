#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 09:08:14 2018

@author: hxw186

MCMC procedure to find the best partition
"""

from tract import Tract
from community_area import CommunityArea
from regression import Linear_regression_evaluation, Linear_regression_training
import random
from math import exp
import numpy as np
from shapely.ops import cascaded_union
import matplotlib.pyplot as plt


def initialize():
    global M, T, featureName, targetName, CA_maxsize, mae1, cnt, iter_cnt, \
        mae_series, mae_index
    print "# initialize"
    random.seed(0)
    Tract.createAllTracts()
    CommunityArea.createAllCAs(Tract.tracts)
    featureName = Tract.income_description.keys()[:5]
    targetName = 'total'
    M = 100
    T = 10
    CA_maxsize = 30
    mae1, _, _ = Linear_regression_training(CommunityArea.features, featureName, targetName)
    cnt = 0
    iter_cnt = 0
    mae_series = [mae1]
    mae_index = [0]


def F(ae):
    return exp(-ae / T)


def convergence_plot():
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(mae_index, mae_series)
    plt.xlabel("sample index")
    plt.ylabel("Training errors")
    
    plt.subplot(1,2,2)
    plt.plot(mae_series)
    plt.xlabel("iterations")
    plt.ylabel("Training errors")


def MCMC_sampling(sample_func, update_sample_weight_func):
    """
    MCMC search for optimal solution.
    Input:
        sample_func is the sample proposal method.
    Output:
        Optimal partition plot and training error decreasing trend.
    """
    global mae1, cnt, iter_cnt
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
        
        # evaluate new partition
        mae2, _, _ = Linear_regression_training(CommunityArea.features, featureName, targetName)
        update_sample_weight_func(mae1, mae2, t)
        gamma = F(mae2) / F(mae1)
        sr = random.random()
        
        if sr < gamma: # made progress
            mae_series.append(mae2)
            mae_index.append(iter_cnt)
            print "{} --> {} in {} steps".format(mae1, mae2, cnt)
            mae1 = mae2
        
            # update tract boundary set for next round sampling 
            Tract.updateBoundarySet(t)
            cnt = 0 # reset counter
            
            if len(mae_series) > 50 and np.std(mae_series[-50:]) < 20:
                # when mae converges
                CommunityArea.visualizeCAs(fname="CAs-iter-final.png")
                break
            if iter_cnt % 500 == 0:
                CommunityArea.visualizeCAs(fname="CAs-iter-{}.png".format(iter_cnt))
        else:
            # restore communities features
            t.CA = prv_caid
            CommunityArea.updateCAFeatures(t, new_caid, prv_caid)


def naive_MCMC():
    initialize()
    MCMC_sampling(random.sample, lambda ae1, ae2, t : 1)
    convergence_plot()


def adaptive_mcmc():
    initialize()
    tract_weight = np.ones(len(Tract.tracts))
    pass
    convergence_plot()


if __name__ == '__main__':
    naive_MCMC()

