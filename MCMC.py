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
import numpy as np
from shapely.ops import cascaded_union
import matplotlib.pyplot as plt


def initialize():
    global M, T, lmbda, featureName, targetName, CA_maxsize, mae1, errors1, cnt, iter_cnt, \
        mae_series, mae_index, var_series,pop_variance1,f_series
    print "# initialize"
    random.seed(0)
    Tract.createAllTracts()
    CommunityArea.createAllCAs(Tract.tracts)
    featureName = CommunityArea.featureNames
    targetName = 'total'
    M = 100
    T = 10
    lmbda = .001
    CA_maxsize = 30
    # Plot original community population distribution
    CommunityArea.visualizePopDist(fname='orig-pop-distribution')
    print "# sampling"
    CA_maxsize = 30
    mae1, _, _,errors1 = NB_regression_training(CommunityArea.features, featureName, targetName)
    pop_variance1 = np.var(CommunityArea.population)
    cnt = 0
    iter_cnt = 0
    mae_series = [mae1]
    var_series = [pop_variance1]
    mae_index = [0]
    f_series = []


def get_f(ae,T,penalty=None,log=True,lmbda=1.0):
    """
    compute the "energy function F".

    :param ae: Error measurement
    :param T: Temperature parameter
    :param penalty: value to penalize constrain object
    :param log: (Bool) Return f on log scale
    :param lmbda: regularization on penalty term
    :return: the 'energy' of a given state
    """
    if penalty is None:
        penalty = 0


    if log:
        return -(ae + lmbda*penalty) / T
    else:
        return np.exp(-(ae + lmbda*penalty) / T)

def get_gamma(f_current,f_proposed,symmetric=True,log=True,q_proposed_given_current=None,q_current_given_proposed=None):
    """
    Compute gamma to be used in acceptance probability of proposed state
    :param f_current: f ("energy function") of current state
    :param f_proposed: f ("energy function") of proposed state
    :param log: (bool) Are f_current and f_proposed on log scale?
    :return: value for gamma, or probability of accepting proposed state
    """
    if symmetric:

        if log:
            alpha = f_proposed - f_current
            gamma = np.min((0,alpha))
        else:
            alpha = f_proposed/f_current
            gamma = np.min((1,alpha))
    else:

        if log:
            alpha = f_proposed - f_current + q_current_given_proposed - q_proposed_given_current
            gamma = np.min((0,alpha))
        else:
            alpha = f_proposed*q_current_given_proposed/f_current*q_proposed_given_current
            gamma = np.min((1,alpha))

    return gamma


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


def softmaxSamplingScheme(errors,community_structure_dict,boundary_tracts,query_ca_prob=None,log=False):
    """
    Function to hierarchicaly sample (1) Communities, (2) tracts within the given community
    :param errors: Vector of errors for softmax
    :param community_structure_dict: Dictionary of community objects -- reflective of conditional states community
    structure. e.g., x'|x  or x|x'
    :param boundary_tracts: List of tracts on boundary, given existing community structure
    :param query_ca_prob:
                    If None: return the probability of randomly selected
                    else: return probability of selecting given tract ID

    :return:
        t: randomly sampled tract (within sampled community
        sample_ca_id: The selected community ID number. If query_ca_prob == None, this is randomly selected. Else return
                query_ca_prob
        sample_ca_prob: Likelihood that sampled community area, i, was sampled:
                i.e., p(CA_i)
        tract_prob: Likelihood (uniform) that tract, j, was sampled conditinal on the sampled community:
                i.e., p(t_j | CA_i)

    """

    # Compute softmax of errors for sampling probabilities
    ca_probs = softmax(errors, log=log)

    if log:
        ca_choice_probs = np.exp(ca_probs)
    else:
        ca_choice_probs = ca_probs


    # Sample community -- probabilities derived from softmax of regression errors
    if query_ca_prob is not None:
        sample_ca_id = query_ca_prob
    else:
        sample_ca_id = np.random.choice(a=community_structure_dict.keys(), size=1, replace=False, p=ca_choice_probs)[0]

    # Collect tracts within sampled community area that lie on community boundary
    sample_ca_boundary_tracts = []
    for tract in boundary_tracts:
        if tract.CA == sample_ca_id:
            sample_ca_boundary_tracts.append(tract)

    # Sample tract (on boundary) within previously sampled community area
    t = np.random.choice(a=sample_ca_boundary_tracts, size=1, replace=False)[0]
    sample_ca_prob = ca_probs.ix[sample_ca_id]

    tract_prob = 1 / float(len(sample_ca_boundary_tracts))
    if log:
        tract_prob = np.log(tract_prob)

    return t, sample_ca_id, sample_ca_prob, tract_prob


def plotMcmcDiagnostics(mae_index,error_array,f_array,variance_array,fname='mcmc-diagnostics'):
    #x = range(len(error_array))
    # Two subplots, the axes array is 1-d
    f, axarr = plt.subplots(3, sharex=True,figsize=(12,8))
    axarr[0].plot(mae_index, np.array(error_array))
    axarr[0].set_title('Mean Absolute Error')
    axarr[1].plot(mae_index, np.array(variance_array))
    axarr[1].set_title('Population Variance (over communities)')
    axarr[2].plot(mae_index, f_array)
    axarr[2].set_title('f - lambda = {}'.format(lmbda))

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
        mae2, _, _, _ = NB_regression_training(CommunityArea.features, featureName, targetName)
        # Calculate acceptance probability --> Put on log scale
        # calculate f ('energy') of current and proposed states
        f_current = get_f(ae = mae1, T=T,penalty=pop_variance1,log=True,lmbda=lmbda)
        f_proposed = get_f(ae = mae2, T=T,penalty=pop_variance2,log=True,lmbda=lmbda)
        # Compute gamma for acceptance probability
        gamma = get_gamma(f_current=f_current,f_proposed=f_proposed,log=True)
        # Generate random number on log scale
        sr = np.log(random.random())
        update_sample_weight_func(mae1, mae2, t)

        if sr < gamma: # made progress
            mae_series.append(mae2)
            var_series.append(pop_variance2)
            print "Iteration {}: {} --> {} in {} steps".format(iter_cnt, mae1, mae2, cnt)
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
                plotMcmcDiagnostics(mae_index=mae_index,
                                    error_array=mae_series,
                                    variance_array=var_series,
                                    fname='mcmc-diagnostics-{}'.format(iter_cnt))

        else:
            # restore communities features
            t.CA = prv_caid
            CommunityArea.updateCAFeatures(t, new_caid, prv_caid)


def mcmcSamplerSoftmax(project_name):
    """
    MCMC search for optimal solution.
    Input:
        sample_func is the sample proposal method.
        update_sample_weight_func updates sampling auxilary variables.
    Output:
        Optimal partition plot and training error decreasing trend.
    """
    global mae1, errors1, cnt, iter_cnt, pop_variance1
    print "# sampling"
    while cnt <= M:
        cnt += 1
        iter_cnt += 1


        ## Hierearchicaly sample community, then boundary tract within community

        t, sample_ca_id ,log_sample_ca_prob, log_tract_prob = softmaxSamplingScheme(errors=errors1,
                                                                            community_structure_dict=CommunityArea.CAs,
                                                                            boundary_tracts=Tract.boundarySet,
                                                                            log=True)

        """
        # Monitor one tract for debugging
        if iter_cnt == 1:
            test_t = t
            test_t_id = test_t.id
            test_orig_ca_id = test_t.CA

        test_t_updated = Tract.tracts[test_t_id]
        test_t_updated_ca = test_t_updated.CA
        """

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

        # Update current state to proposed state
        t.CA = new_caid
        CommunityArea.updateCAFeatures(t, prv_caid, new_caid)
        Tract.updateBoundarySet(t)
        # Get updated variance of population distribution
        pop_variance2 = np.var(CommunityArea.population)
        # evaluate new partition
        mae2, _, _,errors2 = NB_regression_training(CommunityArea.features, featureName, targetName)
        # Calculate acceptance probability --> Put on log scale
        # calculate f ('energy') of current and proposed states
        f_current = get_f(ae=mae1, T=T, penalty=pop_variance1, log=True,lmbda=lmbda)
        f_proposed = get_f(ae=mae2, T=T, penalty=pop_variance2, log=True,lmbda=lmbda)

        if iter_cnt == 1:
            # Initialize f series
            f_series.append(f_current)
        # We need to compute Q to get gamma, since Q is non-symmetric under the softmax sampling scheme
        log_q_proposed_given_current = log_sample_ca_prob + log_tract_prob

        # Reverse conditioning to get q(z | z'); i.e., probability of current state given proposed state
        _, _, log_sample_ca_prob_reverse, log_tract_prob_reverse = softmaxSamplingScheme(errors=errors2,
                                                                                 community_structure_dict=CommunityArea.CAs,
                                                                                 boundary_tracts=Tract.boundarySet,
                                                                                 query_ca_prob=prv_caid,
                                                                                 log=True)
        log_q_current_given_proposed = log_sample_ca_prob_reverse + log_tract_prob_reverse

        # Compute gamma for acceptance probability
        gamma = get_gamma(f_current=f_current,
                          f_proposed=f_proposed,
                          log=True,
                          symmetric=False,
                          q_current_given_proposed=log_q_current_given_proposed,
                          q_proposed_given_current=log_q_proposed_given_current)

        # Generate random number on log scale
        sr = np.log(random.random())


        if sr < gamma:  # made progress
            mae_series.append(mae2)
            var_series.append(pop_variance2)
            f_series.append(f_proposed)
            print "Iteration {}: {} --> {} in {} steps".format(iter_cnt, mae1, mae2, cnt)
            # Update error, variance
            mae1, pop_variance1,errors1 = mae2, pop_variance2,errors2
            mae_index.append(iter_cnt)

            # update tract boundary set for next round sampling

            cnt = 0  # reset counter

            if len(f_series) > 100 and np.std(f_series[-25:]) < 3:
                # when mae converges
                print "converge in {} samples with {} acceptances \
                    sample conversion rate {}".format(iter_cnt, len(mae_series),
                                                      len(mae_series) / float(iter_cnt))
                CommunityArea.visualizeCAs(fname=project_name+"-CAs-iter-final.png")
                CommunityArea.visualizePopDist(fname=project_name+'-final-pop-distribution')

                break

        else:
            # restore community-tract structure to original state
            t.CA = prv_caid
            CommunityArea.updateCAFeatures(t, new_caid, prv_caid)
            Tract.updateBoundarySet(t)

        if iter_cnt % 100 == 0:
            CommunityArea.visualizeCAs(fname=project_name+"-CAs-iter-{}.png".format(iter_cnt))
            CommunityArea.visualizePopDist(fname=project_name+'-pop-distribution-iter-{}'.format(iter_cnt))
            plotMcmcDiagnostics(mae_index=mae_index,
                                error_array=mae_series,
                                variance_array=var_series,
                                f_array = f_series,
                                fname=project_name+'-mcmc-diagnostics-{}'.format(iter_cnt))

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


def MCMC_softmax_proposal(project_name):
    initialize()
    # loo evaluation test data on original boundary
    leaveOneOut_evaluation(2011, "Administrative boundary")
    # restore training data
    CommunityArea._initializeCAfeatures(2010)

    mcmcSamplerSoftmax(project_name)
    plotMcmcDiagnostics(mae_index=mae_index,
                        error_array=mae_series,
                        f_array=f_series,
                        variance_array=var_series,
                        fname=project_name+"-mcmc-diagnostics-final")
    leaveOneOut_evaluation(2011)



if __name__ == '__main__':
    MCMC_softmax_proposal('variance-penalty')


