"""
Created on Dec 11, 2017. HJ

Instantiate tract boundary from shapefile
"""


import matplotlib
matplotlib.rc('pdf', fonttype=42)
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box
import shapefile
from feature_utils import retrieve_income_features, retrieve_crime_count


class Tract:

    def __init__(self, tid, shp, rec=None):
        """Build one tract from the shapefile._Shape object"""
        self.id = tid
        self.bbox = box(*shp.bbox)
        self.polygon = Polygon(shp.points)
        self.centroid = (self.polygon.centroid.x, self.polygon.centroid.y)
        self.count = {'total': 0} # type: value
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
    def visualizeTracts(cls, tracts=None):
        if tracts == None:
            tracts = cls.tracts
        from descartes import PolygonPatch
        f = plt.figure(figsize=(16, 16))
        ax = f.gca()
        for _, t in tracts.items():
            ax.add_patch(PolygonPatch(t.polygon, alpha=0.5, fc="green"))
        ax.axis("scaled")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig("tracts.png")

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
        cls.features = f.join(y)
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
#    t1, t2 = compare_tract_shapefiles()
    trts0 = Tract.createAllTracts()
    Tract.visualizeTracts()
    Tract.visualizeTractsAdjacency()
