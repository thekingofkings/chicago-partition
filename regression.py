#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 16:50:32 2017

@author: kok

Negative Binomial (NB) regression model.
"""

import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import LeaveOneOut
from sklearn import linear_model

def computeError(residual,metric='mse', y_true=None, precision=2):

    #esidual = y_true - y_hat

    if metric == 'mse':
        error_metric = np.mean(np.power(residual,2))
    elif metric == 'sse':
        error_metric = np.sum(np.power(residual, 2))
    elif metric == 'mae':
        error_metric = np.mean(np.abs(residual))
    elif metric == 'mre' and y_true is not None:
        error_metric = np.sum(np.abs(residual)) / np.sum(y_true)
    elif metric == 'rmse':
        error_metric = np.power(np.mean(np.power(residual, 2)),.5)
    else:
        raise Exception("error metric must be mse, rmse, sse, or mre")

    return np.round(error_metric,precision)


def NB_regression_evaluation(df, featureNames, targetName):
    """
    Use python statsmodels lib to evaluate NB regression.
    The evaluation setting is leave-one-out.

    Input:
        The pandas.DataFrame of CA level features
    Output:
        The mean error of multi-rounds leave-one-out.
            - mae
            - mse
            - mre
    """
    df.dropna(inplace=True)
    errors = []
    loo = LeaveOneOut()
    features = df[featureNames]
#    standardScaler = preprocessing.StandardScaler()
#    features = standardScaler.fit_transform(features_raw)
    crimeRate = df[targetName]

    # TODO: Delete line
    feature_names_all = list(df.columns)

    target_in_names = targetName in feature_names_all

    for train_idx, test_idx in loo.split(df):
        X_train, y_train = features.iloc[train_idx], crimeRate.iloc[train_idx]
        X_test, y_test = features.iloc[test_idx], crimeRate.iloc[test_idx]
        nbmodel = sm.GLM(y_train, X_train, family=sm.families.NegativeBinomial())
        model_res = nbmodel.fit()
        y_pred = nbmodel.predict(model_res.params, X_test)
        errors.append(abs(y_pred[0] - y_test.iat[0]))

    rmse = computeError(residual=errors,metric='rmse')
    mae = computeError(residual=errors,metric='mae')
    mre = computeError(residual=errors,metric='mre',y_true=crimeRate)

    return mae, rmse, mre


def NB_regression_training(df, featureNames, targetName):
    """
    NB training for partition search
    """
    df.dropna(inplace=True)
    crimeRate = df[targetName]
    nbmodel = sm.GLM(crimeRate, df[featureNames], family=sm.families.NegativeBinomial())
    model_res = nbmodel.fit()
    betas = model_res.params
    y_pred = nbmodel.predict(model_res.params, df[featureNames])
    errors = abs(crimeRate - y_pred)
    #rel_errors = errors / np.max(errors)
    # Transform errors to standard normal scale (subtract mean, divide by standard deviation)
    rel_errors = (errors - np.mean(errors))/np.std(errors)
    return np.mean(errors), np.std(errors), np.mean(errors)/np.mean(crimeRate), rel_errors, betas


def test_NB_regression():
    """
    Test function of NB model.
    """
    from tract import Tract
    from community_area import CommunityArea
    Tract.createAllTracts()
    CommunityArea.createAllCAs(Tract.tracts)
    featureName = CommunityArea.featureNames
    targetName = 'total'
    print NB_regression_evaluation(CommunityArea.features, featureName, targetName)
    print NB_regression_training(CommunityArea.features, featureName, targetName)



def Linear_regression_evaluation(df, featureNames, targetName):
    """
    Use sklearn linear_model to evaluate LR model (with leave one out).
    """
    errors = []
    loo = LeaveOneOut()
    features = df[featureNames]
    crimeRate = df[targetName]
    for train_idx, test_idx in loo.split(df):
        X_train, y_train = features.iloc[train_idx], crimeRate.iloc[train_idx]
        X_test, y_test = features.iloc[test_idx], crimeRate.iloc[test_idx]
        lrmodel = linear_model.LinearRegression()
        lr_res = lrmodel.fit(X_train, y_train)
        y_pred = lr_res.predict(X_test)
        errors.append(abs(y_pred[0] - y_test.iat[0]))
    return np.mean(errors), np.std(errors), np.mean(errors)/np.mean(crimeRate)


def Linear_regression_training(df, featureNames, targetName):
    """
    Use sklearn linear_model to train LR model (training to find best partition)
    """
    crimeRate = df[targetName]
    lrmodel = linear_model.LinearRegression()
    lr_res = lrmodel.fit(df[featureNames], crimeRate)
    y_pred = lr_res.predict(df[featureNames])
    errors = abs(y_pred - crimeRate)
    return np.mean(errors), np.std(errors), np.mean(errors)/np.mean(crimeRate)


def test_LR_regression():
    """
    Test function for LR model.
    """
    from tract import Tract
    from community_area import CommunityArea
    Tract.createAllTracts()
    CommunityArea.createAllCAs(Tract.tracts)
    featureName = CommunityArea.featureNames
    targetName = 'total'
    print Linear_regression_evaluation(CommunityArea.features, featureName, targetName)
    print Linear_regression_training(CommunityArea.features, featureName, targetName)


if __name__ == '__main__':
    test_NB_regression()
    test_LR_regression()
    