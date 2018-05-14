"""
Created on Dec 11, 2017. HJ

Instantiate tract boundary from shapefile
"""


import matplotlib
matplotlib.rc('pdf', fonttype=42)
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box
import shapefile
from feature_utils import *
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering, spectral_clustering
import numpy as np


class Tract:
    """
    Define one tract.

    A tract is a census spatial unit with roughly 2000 populations. We use tract
    as the minimum unit, and later build Community Area (CA) on top of tract. 
    For each tract we collect related urban features.

    Instance Attributes
    -------------------
    tid : int32
        The tract ID as a integer, e.g. 17031690900.
    polygon : shapely.geometry.Polygon
        The boundary coordinates of a tract.
    CA : int32
        The CA assignment of this tract.
    neighbors : list
        A list of tract instances that are adjacent.
    onEdge : boolean
        Whether current tract is on CA boundary.

    Class Attributes
    ----------------
    tracts : dict
        A dictionary of all tracts. Key is tract ID. Value is tract instance.
    tract_index : list
        All tract IDs in a list sorted in ascending order.
    features : pandas.DataFrame
        All tract features in a dataframe.
    featureNames: list
        The list of column names that will be used as predictor (X).
    boundarySet : set
        A set of tracts CA boundary given current partition
    """

    def __init__(self, tid, shp, rec=None):
        """Build one tract from the shapefile._Shape object"""
        self.id = tid
        self.bbox = box(*shp.bbox)
        self.polygon = Polygon(shp.points)
        self.centroid = (self.polygon.centroid.x, self.polygon.centroid.y)
        if rec != None:
            self.CA = int(rec[6])
        else:
            self.CA = None
        # for adjacency information
        self.neighbors = []
        self.onEdge = False

    @classmethod
    def createAllTracts(cls, fname="data/Census-Tracts-2010/chicago-tract", 
                        calculateAdjacency=True):
        cls.sf = shapefile.Reader(fname)
        tracts = {}
        shps = cls.sf.shapes()
        for idx, shp in enumerate(shps):
            rec = cls.sf.record(idx)
            tid = int("".join([rec[0], rec[1], rec[2]]))
            trt = Tract(tid, shp, rec)
            tracts[tid] = trt
        cls.tracts = tracts
        # sorted index of all tract IDs
        cls.tract_index = sorted(cls.tracts.keys())
        # calculate spatial adjacency graph
        if calculateAdjacency:
            cls.spatialAdjacency()
        return tracts

    @classmethod
    def visualizeTracts(cls, tractIDs=None, tractColors=None, fsize=(16,16), fname="tracts.png",labels=False):
        tracts = {}
        if tractIDs == None:
            tracts = cls.tracts
        else:
            for tid in tractIDs:
                tracts[tid] = cls.tracts[tid]
        if tractColors == None:
            tractColors = dict(zip(tracts.keys(), ['green']* len(tracts)))
        print tractColors
        from descartes import PolygonPatch
        f = plt.figure(figsize=fsize)
        ax = f.gca()
        for k, t in tracts.items():
            ax.add_patch(PolygonPatch(t.polygon, alpha=0.5, fc=tractColors[k]))
            if labels:
                ax.text(t.polygon.centroid.x,
                        t.polygon.centroid.y,
                        int(t.id),
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=8)
        ax.axis("scaled")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(fname)

    @classmethod
    def generateFeatures(cls, crimeYear=2010):
        """
        Generate one feature matrices for all tracts.
        Output:
            Tract.features - a dataframe of features. The header is feature name.
        """
        cls.crimeYear = crimeYear
        f, cls.income_description = retrieve_income_features()
        y = retrieve_crime_count(crimeYear)
        price_f = retrieve_house_price_features()
        # get POI features
        # POI DataFrame rows are sorted by tract ID in ascending order
        poi_f = retrieve_POI_features()
        poi_f.index = sorted(cls.tract_index)
        cls.features = f.join([y, price_f, poi_f])
        cls.featureNames = list(poi_f.columns)
        cols = list(cls.features.columns)
        return cls.features

    @classmethod
    def spatialAdjacency(cls):
        """
        Calculate the adjacent tracts.

        Notice that `shapely.touches` return True if there is one point touch.
        """
        for focalKey, focalTract in cls.tracts.items():
            for otherKey, otherTract in cls.tracts.items():
                if otherKey != focalKey and focalTract.polygon.touches(otherTract.polygon):
                    intersec = focalTract.polygon.intersection(otherTract.polygon)
                    if intersec.geom_type != 'Point':
                        focalTract.neighbors.append(otherTract)
        # calculate whether the tract is on CA boundary
        cls.initializeBoundarySet()

    @classmethod
    def initializeBoundarySet(cls):
        """
        Initialize the boundary set on given partitions.
        """
        cls.boundarySet = set()
        for _, t in cls.tracts.items():
            for n in t.neighbors:
                if t.CA != n.CA:
                    t.onEdge = True
                    cls.boundarySet.add(t)
                    break

    @classmethod
    def updateBoundarySet(cls, tract):
        """
        Update bounary set for next round sampling
        """
        tracts_check = [tract] + tract.neighbors
        for t in tracts_check:
            onEdge = False
            for n in t.neighbors:
                if t.CA != n.CA:
                    onEdge = True
                    break
            if not onEdge:
                if t.onEdge:
                    t.onEdge = False
                    cls.boundarySet.remove(t)
            else:
                t.onEdge = True
                cls.boundarySet.add(t)

    @classmethod
    def visualizeTractsAdjacency(cls):
        """
        Plot tract adjacency graph. Each tract is ploted with its centroid.
        The adjacency 
        """
        from matplotlib.lines import Line2D
        tracts = cls.tracts
        f = plt.figure(figsize=(16, 16))
        ax = f.gca()
        for _, t in tracts.items():
            for n in t.neighbors:
                ax.add_line(Line2D(*zip(t.centroid, n.centroid)))
        ax.axis('scaled')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig("adjacency.png")


    @classmethod
    def getPartition(cls):
        return [cls.tracts[k].CA for k in cls.tract_index]

    @classmethod
    def getTractPosID(cls, t):
        return cls.tract_index.index(t.id)

    @classmethod
    def restorePartition(cls, partition):
        for i, k in enumerate(cls.tract_index):
            cls.tracts[k].CA = partition[i]

    @classmethod
    def writePartition(cls,fname):
        f = open("output/" + fname,'w')
        tract_ca_assignment = cls.getPartition()

        for i in tract_ca_assignment:
            f.write(str(i))
            f.write("\n")
        f.close()


    @classmethod
    def readPartition(cls,fname):
        f = open("output/" + fname, 'r')
        partition = f.readlines()
        partition_clean = [int(x.rstrip()) for x in partition]
        f.close()
        cls.restorePartition(partition=partition_clean)

    @classmethod
    def agglomerativeClustering(cls, cluster_X=True,cluster_y=False,y=None, algorithm = "ward"):
        '''
        using agglomerative clustering
        :return: tract to CA mapping
        '''
        connectivity, node_value, CA_count, tract_ids = cls.constructConnectivity(income_features=cluster_X,
                                                                                  target_bool=cluster_y,
                                                                                  target=y)


        if algorithm == "ward":
            ward = AgglomerativeClustering(n_clusters=CA_count, linkage="ward",
                                           connectivity=connectivity)
        elif algorithm =="average_cosine":
            ward = AgglomerativeClustering(n_clusters=CA_count, linkage="average",
                                           affinity="cosine",
                                           connectivity=connectivity)
        elif algorithm =="average_cityblock":
            ward = AgglomerativeClustering(n_clusters=CA_count, linkage="average",
                                           affinity="cityblock",
                                           connectivity=connectivity)
        elif algorithm =="complete_cosine":
            ward = AgglomerativeClustering(n_clusters=CA_count, linkage="complete",
                                           affinity="cosine",
                                           connectivity=connectivity)
        else:
            raise "ERROR: agglomerative clustering params wrong!"
        ward.fit(node_value)
        labels = ward.labels_
        tract_to_CA_dict = dict(zip(tract_ids,labels))
        cls.updateCA(tract_to_CA_dict)

    @classmethod
    def kMeansClustering(cls,cluster_X=True,cluster_y=False,y=None):
        """
        cluster tracts using kmeans clustering
        :return:
        """
        connectivity, node_value, CA_count, tract_ids = cls.constructConnectivity(income_features=cluster_X,
                                                                                  target_bool=cluster_y,
                                                                                  target=y)

        km = KMeans(n_clusters=CA_count,init='k-means++')
        km.fit(node_value)
        labels = km.labels_
        tract_to_CA_dict = dict(zip(tract_ids, labels))
        cls.updateCA(tract_to_CA_dict)
        return tract_to_CA_dict

    @classmethod
    def spectralClustering(cls,cluster_X=True,cluster_y=False,y=None,assign_labels='discretize'):
        """
        cluster tracts using spectral clustering
        :return:
        """

        connectivity, node_value, CA_count, tract_ids = cls.constructConnectivity(income_features=cluster_X,
                                                                                  target_bool=cluster_y,
                                                                                  target=y)

        labels = spectral_clustering(connectivity, n_clusters=CA_count,
                                     assign_labels=assign_labels, random_state=None)
        tract_to_CA_dict = dict(zip(tract_ids, labels))
        cls.updateCA(tract_to_CA_dict)
        return tract_to_CA_dict


    @classmethod
    def constructConnectivity(cls,income_features=True,target_bool=False,target=None):
        '''
        Construct connectivity matrix for clustering methods
        :return: Adjacency matrix, node value matrix, number of CA_ids,
                 tract_ids order in cls.items
        '''
        from scipy import sparse
        N = len(cls.tracts)
        I = []
        J = []
        V = []
        X = []
        CA_ids = []
        tract_ids = []

        node_features = list()

        if income_features:
            node_features += cls.income_description.keys()
        if target_bool and target is not None:
            node_features += [target]

        #TODO: Delete line
        feature_names_all = list(cls.features.columns)

        target_in_names = target in feature_names_all

        for focalKey, focalTract in cls.tracts.items():
            tract_ids.append(focalKey)
            CA_ids.append(focalTract.CA)
        for focalKey, focalTract in cls.tracts.items():
            #X.append(cls.features.loc[focalKey, cls.income_description.keys()])
            X.append(cls.features.loc[focalKey, node_features])
            for neighbor in focalTract.neighbors:
                I.append(tract_ids.index(focalKey))
                J.append(tract_ids.index(neighbor.id))
                V.append(1)

        return sparse.coo_matrix((np.array(V), (np.array(I), np.array(J))), shape=(N, N)), \
               np.array(X), np.unique(np.array(CA_ids)).size ,tract_ids

    @classmethod
    def updateCA(cls,tract_to_CA_dict):
        '''
        Update the CA id according to tract_CA mapping
        :param tract_to_CA_dict:
        :return:
        '''
        for focalKey, focalTract in cls.tracts.items():
            focalTract.CA = tract_to_CA_dict[focalKey]


def compare_tract_shapefiles():
    """There are two version of tract level shapfiles.
    Are they the same?"""
    trts1 = Tract.createAllTracts()
    trts2 = Tract.createAllTracts("data/chicago-shp-2010-gps/chicago_tract_wgs84")
    exactly_same = True
    print len(trts1), len(trts2)
    for tid in trts1:
        if tid not in trts2:
            exactly_same = False
            print "{} not in tracts2".format(tid)
            break

        t1 = trts1[tid]
        t2 = trts2[tid]
        if t1.CA != t2.CA:
            exactly_same = False
            print "{} and {} not equal".format(t1.CA, t2.CA)
            break

        x1, y1 = t1.polygon.boundary.coords.xy
        x2, y2 = t2.polygon.boundary.coords.xy
        if len(x1) != len(x2) or len(y1) != len(y2):
            exactly_same = False
            print "length not match {} {} {} {}".format(len(x1), len(x2), len(y1), len(y2))
            break
        else:
            for i, _ in enumerate(x1):
                assert abs(x1[i] - x2[i]) <= 0.00001

    assert exactly_same is True
    print "Good news: two shapfiles are mostly identical."
    return trts1, trts2




if __name__ == '__main__':
    #t1, t2 = compare_tract_shapefiles()
    trts0 = Tract.createAllTracts()
    #Tract.visualizeTracts()
    #Tract.visualizeTractsAdjacency)
    trts0_features = Tract.generateFeatures(crimeYear=2010)

    #tract_to_CA_dict = Tract.agglomerativeClustering()
    # Tract.updateCA(tract_to_CA_dict)
    # trts1 = Tract.tracts


    #tract_to_ca_dict = Tract.kMeansClustering()
    tract_to_ca_dict = Tract.spectralClustering()


