#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 09:08:14 2018

@author: hxw186

MCMC procedure to find the best partition
"""

from tract import Tract
from communityArea import CommunityArea
from regression import Linear_regression_evaluation, Linear_regression_training
import random
from math import exp
import numpy as np
from shapely.ops import cascaded_union
from mcmcUtils import plotMcmcDiagnostics


if __name__ == '__main__':
    print "# initialize"
    random.seed(0)
    Tract.createAllTracts()
    CommunityArea.createAllCAs(Tract.tracts)
    featureName = Tract.income_description.keys()[:5]
    targetName = 'total'
    M = 100
    T = 10
    CA_maxsize = 30
    mae1, std_ae1, mre1 = Linear_regression_training(CommunityArea.features, featureName, targetName)
    pop_variance1 = np.var(CommunityArea.population)

    # Plot original community population distribution
    CommunityArea.visualizePopDist(fname='orig-pop-distribution')
    print "# sampling"
    cnt = 0
    iter_cnt = 0
    mae_series = [mae1]
    var_series = [pop_variance1]
    while cnt <= M:
        cnt += 1
        iter_cnt += 1
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
        mae2, std_ae2, mre2 = Linear_regression_training(CommunityArea.features, featureName, targetName)
        # Calculate acceptance probability --> Put on log scale
        #f1 = exp(- (mae1 + pop_variance1) / T)
        #f2 = exp(- (mae2 + pop_variance2)/ T)
        #gamma = f2 / f1

        gamma = (mae1 - mae2 + pop_variance1 - pop_variance2)/T

        sr = np.log(random.random())
        
        if sr < gamma: # made progress
            mae_series.append(mae2)
            var_series.append(pop_variance2)
            print "Iteration {}: {} --> {} in {} steps".format(iter_cnt,mae1, mae2, cnt)
            mae1, std_ae1, mre1, pop_variance1 = mae2, std_ae2, mre2, pop_variance2
        
            # update tract boundary set for next round sampling 
            Tract.updateBoundarySet(t)
            cnt = 0 # reset counter
            
            if len(mae_series) > 100 and np.std(mae_series[-50:]) < 3:
                # when mae converges
                CommunityArea.visualizeCAs(fname="CAs-iter-final.png")
                CommunityArea.visualizePopDist(fname='final-pop-distribution')
                plotMcmcDiagnostics(error_array=mae_series,variance_array=var_series)

                break
            if iter_cnt % 500 == 0:
                CommunityArea.visualizeCAs(fname="CAs-iter-{}.png".format(iter_cnt))
                CommunityArea.visualizePopDist(fname='pop-distribution-iter-{}'.format(iter_cnt))
        else:
            # restore communities features
            t.CA = prv_caid
            CommunityArea.updateCAFeatures(t, new_caid, prv_caid)

