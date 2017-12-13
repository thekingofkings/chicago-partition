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
        self.features = tract_features.sum(axis=0)
        
        
        
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
        
        # initialize features
        cls.features_raw = Tract.features if hasattr(Tract, "features") else Tract.generateFeatures()
        features_ca = {}
        for ca in CAs.values():
            ca.initializeField()
            features_ca[ca.id] = ca.features
        cls.features = pd.DataFrame.from_dict(data=features_ca, orient="index")
        cls.CAs = CAs
        return CAs
            
        
    @classmethod
    def visualizeCAs(cls, CAs=None):
        if CAs == None:
            CAs = cls.CAs
        from descartes import PolygonPatch
        f = plt.figure(figsize=(6,6))
        ax = f.gca()
        for k, t in CAs.items():
            ax.add_patch(PolygonPatch(t.polygon, alpha=0.5, fc="green"))
        ax.axis("scaled")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig("CAs.png")
        


if __name__ == '__main__':
    Tract.createAllTracts()
    CommunityArea.createAllCAs(Tract.tracts)
    CommunityArea.visualizeCAs()
    