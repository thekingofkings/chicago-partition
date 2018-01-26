#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 07:57:36 2018

@author: hj

Q-learning as adaptive MCMC method

"""
from mcmcSummaries import writeSimulationOutput, plotMcmcDiagnostics
from tract import Tract
from community_area import CommunityArea
from regression import NB_regression_training
from shapely.ops import cascaded_union
import random
import numpy as np
import math
from MCMC import leaveOneOut_evaluation, get_f,isConvergent
from keras.layers import Input, Embedding, Dense, concatenate, Flatten
from keras.models import Model 
from keras.callbacks import TensorBoard




def initialize(project_name, targetName):
    global featureName, M, T, CA_maxsize, mae1, mae_series, mae_index, \
        iter_cnt, F_series, pop_std1, std_series, cnt, epsilon
    print "# initialize"
    random.seed(0)
    epsilon = {"acc_len": 100, "prev_len": 50, "f_sd": 1.5}
    Tract.createAllTracts()
    CommunityArea.createAllCAs(Tract.tracts)
    featureName = CommunityArea.featureNames

    M = 100
    T = 10
    CA_maxsize = 30
    mae1, _, _, errors = NB_regression_training(CommunityArea.features, featureName, targetName)
    pop_std1 = np.std(CommunityArea.population)
    iter_cnt = 0
    cnt = 0
    mae_series = [mae1]
    std_series = [pop_std1]
    mae_index = [0]
    F_series = [get_f(mae1, T, penalty=pop_std1)]



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
    return (t, prv_caid, new_caid)


def q_learning(project_name,targetName='total'):
    global iter_cnt, mae_series, F_series, pop_std1, cnt, mae1
    initialize()
    
    # loo evaluation test data on original boundary
#    leaveOneOut_evaluation(2011, "Administrative boundary")
    # restore training data
#    CommunityArea._initializeCAfeatures(2010)
    
    partition = Input(shape=(801,), dtype='int32', name='partition')
    action_tract = Input(shape=(1,), dtype='int32', name='action_target_tract')
    action_toCA = Input(shape=(1,), dtype='int32', name='action_new_CA')
    
    ca_embed = Flatten()(Embedding(77, 2, input_length=802)(concatenate([partition, action_toCA])))
    tract_embed = Flatten()(Embedding(801, 4, input_length=1)(action_tract))
    
    x = concatenate([ca_embed, tract_embed])
    x = Dense(200, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    
    output = Dense(1, activation='sigmoid', name='delta_error')(x)
    
    model = Model(inputs=[partition, action_tract, action_toCA], outputs=[output])
    model.compile(optimizer='rmsprop', loss='mse')
    
    tbCallback = TensorBoard(log_dir="/tmp/tensorboard_logs", batch_size=32, write_graph=True, 
                             write_grads=False, write_images=False, embeddings_freq=0,
                             embeddings_layer_names=None, embeddings_metadata=None)


    print "# sampling"
    while True:
        iter_cnt += 1
        
        i = 0
        action_tracts = []
        action_toCAs = []
        partitions = []
        gains = []
        curPartition = Tract.getPartition()

        F_cur = get_f(ae=mae1, T=T, penalty=pop_std1)
        # random sample a batch for Q-learning
        while i < 32:
            state = Tract.getPartition()
            sample_res = sample_once()
            if sample_res == None:
                continue

            t, prv_caid, new_caid = sample_res
            t.CA = new_caid
            # update communities features for evaluation
            CommunityArea.updateCAFeatures(*sample_res)
            # Get updated variance of population distribution
            pop_std2 = np.std(CommunityArea.population)
            # evaluate new partition
            mae2, _, _, _ = NB_regression_training(CommunityArea.features, featureName, targetName)
            # Calculate acceptance probability --> Put on log scale
            # calculate f ('energy') of current and proposed states
            F_next = get_f(ae = mae2, T=T, penalty=pop_std2)
            gain = 1 / (1 + math.exp(- F_next + F_cur))

            partitions.append(state)
            action_tracts.append(Tract.getTractPosID(t))
            action_toCAs.append(new_caid-1)
            gains.append(gain)

            Tract.updateBoundarySet(t)
            i += 1
            cnt += 1
            F_cur = F_next

#        print "fit model. samples gains min {}, mean {}, max {}".format(np.min(gains), np.mean(gains), np.max(gains))
        model.fit(x={'partition': np.array(partitions)-1, 
                     'action_target_tract': np.array(action_tracts), 
                     'action_new_CA': np.array(action_toCAs)}, 
                    y=np.array(gains),
                    epochs=2,
                    callbacks=[tbCallback])

        # reset the permutation of partitions
        Tract.restorePartition(curPartition)
        Tract.initializeBoundarySet()
        CommunityArea.CAs = {}
        CommunityArea.createAllCAs(Tract.tracts)
        
        j = 0
        gain_highest = 0.4
        action_tract = None
        action_ca = None
        gain_preds = []
        while j < 32:
            action = sample_once()
            if action is None:
                continue
            t, prv_caid, new_caid = action
            gain_pred = model.predict(x={'partition': (np.array(curPartition)-1)[None],
                                         'action_target_tract': np.array(Tract.getTractPosID(t))[None],
                                         'action_new_CA': np.array(new_caid-1)[None]})
            gain_preds.append(gain_pred)
            if gain_pred > gain_highest:
                gain_highest = gain_pred
                action_tract = t
                action_ca = new_caid
            
            j += 1
#        print "Q function esimate gain_pred min {}, mean {}, max {}".format(np.min(gain_preds),
#                                                np.mean(gain_preds), np.max(gain_preds))

        # take the best action
        if action_tract is None or action_ca is None:
            print "!== Did not find an action to improve within 32 trials. Restart"
            continue
        else:
            prv_caid = action_tract.CA
            action_tract.CA = action_ca
            Tract.updateBoundarySet(action_tract)
            CommunityArea.updateCAFeatures(action_tract, prv_caid, action_ca)
            # Get updated variance of population distribution
            pop_std2 = np.std(CommunityArea.population)
            # evaluate new partition
            mae2, _, _, _ = NB_regression_training(CommunityArea.features, featureName, targetName)
            mae_series.append(mae2)
            std_series.append(pop_std2)
            F_series.append(get_f(mae2, T, pop_std2))
            print "Iteration {}: {} --> {}".format(iter_cnt, mae1, mae2)
            # Update error, variance
            mae1, pop_std1 = mae2, pop_std2
            mae_index.append(iter_cnt)

            if isConvergent(epsilon, F_series):
                # when mae converges
                print "converge in {} samples with {} acceptances \
                    sample conversion rate {}".format(iter_cnt, len(mae_series),
                                                len(mae_series) / float(iter_cnt))
                CommunityArea.visualizeCAs(fname="CAs-iter-final.png")
                CommunityArea.visualizePopDist(fname='final-pop-distribution')
                mean_test_error, sd_test_error, mean_err_mean_val = leaveOneOut_evaluation(2011, targetName.replace('train', 'test'))
                plotMcmcDiagnostics(iter_cnt, mae_index, mae_series, F_series, std_series,
                                    fname=project_name)
                writeSimulationOutput(project_name=project_name,
                                      error=mean_test_error,
                                      n_iter_conv=iter_cnt,
                                      accept_rate=len(mae_series) / float(iter_cnt))


                Tract.writePartition(fname=project_name + "-final-partition.txt")
                del model
                break

        if iter_cnt % 500 == 0:
            CommunityArea.visualizeCAs(iter_cnt=iter_cnt, fname="CAs-iter-{}.png".format(iter_cnt))
            CommunityArea.visualizePopDist(fname='pop-distribution-iter-{}'.format(iter_cnt),
                                           iter_cnt=iter_cnt)


if __name__ == '__main__':
    for i in range(3, 11):
        project_name = "q-learning-v{}".format(i)
        q_learning(project_name,targetName='total')