#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 09:08:14 2018

@author: hxw186

MCMC procedure to find the best partition
"""

from tract import Tract
from communityArea import CommunityArea
from regression import Linear_regression_evaluation
import random
from math import exp
import numpy as np
from shapely.ops import cascaded_union


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
    mae1, std_ae1, mre1 = Linear_regression_evaluation(CommunityArea.features, featureName, targetName)
    
    print "# sampling"
    cnt = 0
    iter_cnt = 0
    mae_series = [mae1]
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
        
        # evaluate new partition
        mae2, std_ae2, mre2 = Linear_regression_evaluation(CommunityArea.features, featureName, targetName)
        f1 = exp(- mae1 / T)
        f2 = exp(- mae2 / T)
        gamma = f2 / f1
        sr = random.random()
        
        if sr < gamma: # made progress
            mae_series.append(mae2)
            print "{} --> {} in {} steps".format(mae1, mae2, cnt)
            mae1, std_ae1, mre1 = mae2, std_ae2, mre2
        
            # update tract boundary set for next round sampling 
            Tract.updateBoundarySet(t)
            cnt = 0 # reset counter
            
            if len(mae_series) > 50 and np.std(mae_series[-50:]) < 5:
                # when mae converges
                CommunityArea.visualizeCAs(fname="CAs-iter-final.png")
                break
            if iter_cnt % 100 == 0:
                CommunityArea.visualizeCAs(fname="CAs-iter-{}.png".format(iter_cnt))
        else:
            # restore communities features
            t.CA = prv_caid
            CommunityArea.updateCAFeatures(t, new_caid, prv_caid)

