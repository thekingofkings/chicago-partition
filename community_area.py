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
import numpy as np
from feature_utils import retrieve_summarized_income_features
import pylab as P


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
        fdf['train_average_house_price'] = \
                    fdf['train_average_house_price'].replace([np.inf, -np.inf, np.nan], CommunityArea.default_house_price_train)
        fdf['test_average_house_price'] = fdf['test_price'] / fdf['test_count']
        fdf['test_average_house_price'] = \
                    fdf['test_average_house_price'].replace([np.inf, -np.inf, np.nan], CommunityArea.default_house_price_test)
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
        cls.default_house_price_train = np.sum(cls.features_raw['train_price']) / np.sum(cls.features_raw['train_count'])
        cls.default_house_price_test = np.sum(cls.features_raw['test_price']) / np.sum(cls.features_raw['test_count'])
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
    def visualizeCAs(cls,
                     iter_cnt=None,
                     CAs=None,
                     fname="CAs.png",
                     plot_measure=None,
                     labels=False,
                     title=None,
                     case_study=False,
                     comm_to_plot = None,
                     jitter_labels = False):
        """
        Class method to plot community boundaries. Arguments available for plotting a continuous measure over the
        community areas
        :param iter_cnt: (int) Specifies the current iteration 
                            Use if plotting updates in a sequential algorithm; i.e., MCMC. 
                            will be inserted into plot title if title = None
        :param CAs: (dict) Dictionary of communities. If none is given, default is class attribute CAs
        :param fname: (str) name of .png file to be written to disk
        :param plot_measure: (Series) Use if we want to visually scale a continous measure by community are (i.e., heat map)
                             plot_measure is pd.Series instance with the community ID's as the index, and the measure as the
                             values
        :param labels: (bool) Used in conjuction with plot_measure. If True, plot the community ID and measure value at the
                                centroid of each community area
        :param title: (str) Title for plot saved .png. If title = '', then no title, If title is None, then use default
                            title that prints iteration count
        :return: 
        """
        if CAs == None:
            CAs = cls.CAs
        if iter_cnt is None:
            iter_cnt = "completed"

        from descartes import PolygonPatch
        f = plt.figure(figsize=(8,8))
        ax = f.gca()


        if plot_measure is not None:
            # Create heat map
            # Scale colors by sorted values of plot_measure
            col_gradient = np.linspace(0.05, 1, len(plot_measure))
            plot_measure.sort_values(ascending=False,inplace=True)
            # Create color map (dict). Keys: Community ID - Values: Green values (in RGB) in [0,1] interval
            color_map = dict(zip(plot_measure.index,col_gradient))

            if case_study:
                for k, t in CAs.items():
                    if t.id in comm_to_plot:
                        lw = .75
                        ca_id = t.id
                    else:
                        lw = 0
                        ca_id = ''

                    if ca_id == comm_to_plot[0]:
                        P.arrow(-87.6395, 41.661, 0.02, 0.02, fc="k", ec="k",
                                head_width=0.005, head_length=0.01)

                    ax.add_patch(PolygonPatch(t.polygon, alpha=0.85,
                                              fc=(1, color_map[k], 0),
                                              edgecolor='black',
                                              linewidth=lw))

                    if labels:
                        if jitter_labels:
                            x = t.polygon.centroid.x + np.random.uniform(0,.01)
                            y = t.polygon.centroid.y + np.random.uniform(0,.01)
                        else:
                            x = t.polygon.centroid.x
                            y = t.polygon.centroid.y

                        ax.text(x,y,
                                ca_id,
                                horizontalalignment='center',
                                verticalalignment='center', fontsize=18)

            else:
                for k, t in CAs.items():
                    ax.add_patch(PolygonPatch(t.polygon, alpha=0.85,
                                              fc=(1, color_map[k], 0),
                                              edgecolor='black',
                                              linewidth=.5))


                    if labels:
                        ax.text(t.polygon.centroid.x,t.polygon.centroid.y,
                                   t.id,
                                   horizontalalignment='center',
                                   verticalalignment='center', fontsize=18)

        else:

            for k, t in CAs.items():
                ax.add_patch(PolygonPatch(t.polygon, alpha=0.5, fc='green', lw=1.5,edgecolor='black'))

                if labels:
                    # Label plot with community ids at each respective centroid
                    ax.text(t.polygon.centroid.x,
                            t.polygon.centroid.y,
                                int(t.id),
                               horizontalalignment='center',
                               verticalalignment='center', fontsize=12)


        ax.axis("scaled")
        ax.axis("off")
        plt.tight_layout()
        if title is None:
            plt.title('Community Structure -- Iterations: {}'.format(iter_cnt))
        elif title == '':
            pass
        else:
            plt.title(title,fontsize = 20)
        plt.tight_layout()
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


def fig_clustering_baseline(keepBest=True):
    from regression import NB_regression_evaluation

    Tract.createAllTracts()
    Tract.generateFeatures(2011)

    Tract.agglomerativeClustering()
    CommunityArea.createAllCAs(Tract.tracts)
    print NB_regression_evaluation(CommunityArea.features, CommunityArea.featureNames, 'total')
    print NB_regression_evaluation(CommunityArea.features, CommunityArea.featureNames, 'test_average_house_price')
    CommunityArea.visualizeCAs(fname="agg_CAs.pdf", title='')

    if not keepBest:
        Tract.agglomerativeClustering(algorithm="average_cosine")
        CommunityArea.createAllCAs(Tract.tracts)
        print NB_regression_evaluation(CommunityArea.features, CommunityArea.featureNames, 'total')
        print NB_regression_evaluation(CommunityArea.features, CommunityArea.featureNames, 'test_average_house_price')
        CommunityArea.visualizeCAs(fname="agg_CAs_average_cosine.png")
    
        Tract.agglomerativeClustering(algorithm="average_cityblock")
        CommunityArea.createAllCAs(Tract.tracts)
        print NB_regression_evaluation(CommunityArea.features, CommunityArea.featureNames, 'total')
        print NB_regression_evaluation(CommunityArea.features, CommunityArea.featureNames, 'test_average_house_price')
        CommunityArea.visualizeCAs(fname="agg_CAs_average_cityblock.png")
    
        Tract.agglomerativeClustering(algorithm="complete_cosine")
        CommunityArea.createAllCAs(Tract.tracts)
        print NB_regression_evaluation(CommunityArea.features, CommunityArea.featureNames, 'total')
        print NB_regression_evaluation(CommunityArea.features, CommunityArea.featureNames, 'test_average_house_price')
        CommunityArea.visualizeCAs(fname="agg_CAs_complete_cosine.png")

if __name__ == '__main__':
#    Tract.createAllTracts()
#    CommunityArea.createAllCAs(Tract.tracts)
#    CommunityArea.visualizeCAs()

    fig_clustering_baseline()


