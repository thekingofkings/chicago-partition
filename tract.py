"""
Created on Dec 11, 2017. HJ

Instantiate tract boundary from shapefile
"""


import matplotlib
matplotlib.rc('pdf', fonttype=42)
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box
import shapefile
from featureUtils import retrieve_crime_count, retrieve_income_features


class Tract:
    
    def __init__(self, shp, rec=None):
        """Build one tract from the shapefile._Shape object"""
        self.bbox = box(*shp.bbox)
        self.polygon = Polygon(shp.points)
        self.count = {'total': 0} # type: value
        if rec != None:
            self.CA = int(rec[6])
        else:
            self.CA = None
            
    @classmethod
    def createAllTracts(cls, fname="data/Census-Tracts-2010/chicago-tract"):
        cls.sf = shapefile.Reader(fname)
        tracts = {}
        shps = cls.sf.shapes()
        for idx, shp in enumerate(shps):
            rec = cls.sf.record(idx)
            tid = int("".join([rec[0], rec[1], rec[2]]))
            trt = Tract(shp, rec)
            tracts[tid] = trt
        cls.tracts = tracts
        return tracts
    
    @classmethod
    def visualizeTracts(cls, tracts = None):
        if tracts == None:
            tracts = cls.tracts
        from descartes import PolygonPatch
        f = plt.figure(figsize=(6,6))
        ax = f.gca()
        for k, t in tracts.items():
            ax.add_patch(PolygonPatch(t.polygon, alpha=0.5, fc="green"))
        ax.axis("scaled")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig("tracts.png")
        
    @classmethod
    def generateFeatures(cls):
        """
        Generate one feature matrices for all tracts.
        Output:
            Tract.features - a dataframe of features. The header is feature name.
        """
        f, cls.income_description = retrieve_income_features()
        y = retrieve_crime_count()
        cls.features = f.join(y)
        return cls.features



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
            for i in range(len(x1)):
                assert abs(x1[i] - x2[i]) <= 0.00001
        
    assert exactly_same == True
    print "Good news: two shapfiles are mostly identical."
    return trts1, trts2
    
    
if __name__ == '__main__':
    t1, t2 = compare_tract_shapefiles()
    trts1 = Tract.createAllTracts()
    Tract.visualizeTracts()
