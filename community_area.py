#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 09:46:40 2017

@author: kok

Create community areas (CAs) from tracts.
1) merge tracts boundary to get CA boundary.
2) merge tracts features to get CA features.
"""

from tract import Tract
from shapely.ops import cascaded_union
import matplotlib.pyplot as plt
import pandas as pd
from feature_utils import retrieve_summarized_income_features

class CommunityArea:
    
    def __init__(self, caID):
        self.id = caID
        self.tracts = {}
        
    def addTract(self, tID, trct):
        self.tracts[tID] = trct
        
    def initializeField(self):
        """
        Prerequisite:
            all tracts are added to corresponding CA.
        Goal:
            initialize the boundary and features of each CA.
        """
        self.polygon = cascaded_union([e.polygon for e in self.tracts.values()])
        tract_features = CommunityArea.features_raw.loc[ self.tracts.keys() ]
        # merge tract features to get CA features
        # one CA feature is a pandas.DataFrame with one row
        feat_series = tract_features.sum(axis=0)
        feat_vals = feat_series.get_values()[None]
        fdf = pd.DataFrame(feat_vals, columns=feat_series.index.get_values(),
                                     index=[self.id])
        # calculate the average house price (training and testing)
        fdf['train_average_house_price'] = fdf['train_price'] / fdf['train_count']
        fdf['test_average_house_price'] = fdf['test_price'] / fdf['test_count']
        self.features= fdf
        # calculate summarized demo features, such as entropy / percentage 
        if hasattr(CommunityArea, "featureNames"):
            self.features, _ = retrieve_summarized_income_features(self.features)
        else:
            self.features, CommunityArea.featureNames = retrieve_summarized_income_features(self.features)
            #CommunityArea.featureNames += Tract.featureNames


    @classmethod
    def createAllCAs(cls, tracts):
        """
        tracts:
            a dict of Tract, each of which has CA assignment.
        Output:
            a dict of CAs
        """
        CAs = {}
        # initialize boundary
        for tID, trct in tracts.items():
            assert trct.CA != None
            if trct.CA not in CAs:
                ca = CommunityArea(trct.CA)
                ca.addTract(tID, trct)
                CAs[trct.CA] = ca
            else:
                CAs[trct.CA].addTract(tID, trct)
        cls.CAs = CAs
        cls._initializeCAfeatures()
        return CAs


    @classmethod
    def _initializeCAfeatures(cls, crimeYear=2010):
        cls.features_raw = Tract.features if hasattr(Tract, "features") \
            and Tract.crimeYear == crimeYear else Tract.generateFeatures(crimeYear)
        cls.features_ca_dict = {}
        for ca in cls.CAs.values():
            ca.initializeField()
            cls.features_ca_dict[ca.id] = ca.features
        cls.features = pd.concat(cls.features_ca_dict.values())
        # Save population feature for partitioning constraints d
        cls.populationFeature = "B1901001"
        cls.population = cls.features[cls.populationFeature]


    @classmethod
    def updateCAFeatures(cls, tract, prv_CAid, new_CAid):
        """
        Update the CA features, when one tract is flipped from prv_CA to new_CA.
        """
        prv_CA = cls.CAs[prv_CAid]
        del prv_CA.tracts[tract.id]
        prv_CA.initializeField()
        cls.features_ca_dict[prv_CAid] = prv_CA.features
        
        new_CA = cls.CAs[new_CAid]
        new_CA.tracts[tract.id] = tract
        new_CA.initializeField()
        cls.features_ca_dict[new_CAid] = new_CA.features
        
        # convert dict of pandas.Series into DataFrame
        cls.features = pd.concat(cls.features_ca_dict.values())
        cls.population = cls.features[cls.populationFeature]
        

        
    @classmethod
    def visualizeCAs(cls, iter_cnt=None, CAs=None, fname="CAs.png"):
        if CAs == None:
            CAs = cls.CAs
        if iter_cnt is None:
            iter_cnt = "completed"

        from descartes import PolygonPatch
        f = plt.figure(figsize=(6,6))
        ax = f.gca()
        for k, t in CAs.items():
            ax.add_patch(PolygonPatch(t.polygon, alpha=0.5, fc="green"))
        ax.axis("scaled")
        ax.axis("off")
        plt.tight_layout()
        plt.title('Community Structure -- Iterations: {}'.format(iter_cnt))
        plt.savefig("plots/" + fname)
        plt.close()
        plt.clf()

    @classmethod
    def visualizePopDist(cls,fname,iter_cnt=None):
        if iter_cnt is None:
            iter_cnt = "completed"
        pop_df = pd.DataFrame(cls.population)
        pop_df.plot(kind='barh', figsize=(16, 12))
        plt.title('Population Distribution -- Iterations: {}'.format(iter_cnt))
        plt.savefig("plots/" + fname)
        plt.close()
        plt.clf()
        


if __name__ == '__main__':
    Tract.createAllTracts()
    CommunityArea.createAllCAs(Tract.tracts)
#    CommunityArea.visualizeCAs()
    