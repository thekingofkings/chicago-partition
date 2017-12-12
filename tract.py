"""
Created on Dec 11, 2017. HJ

Instantiate tract boundary from shapefile
"""


import matplotlib
matplotlib.rc('pdf', fonttype=42)
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box
import shapefile


class Tract:
    
    def __init__(self, shp, rec=None):
        """Build one tract from the shapefile._Shape object"""
        self.bbox = box(*shp.bbox)
        self.polygon = Polygon(shp.points)
        self.count = {'total': 0} # type: value
        if rec != None:
            self.CA = int(rec[6])
            
    @classmethod
    def createAllTractObjects(cls, fname="data/Census-Tracts-2010/chicago-tract"):
        cls.sf = shapefile.Reader(fname)
        tracts = {}
        shps = cls.sf.shapes()
        for idx, shp in enumerate(shps):
            rec = cls.sf.record(idx)
            tid = rec[5]
            trt = Tract(shp, rec)
            tracts[tid] = trt
        cls.tracts = tracts
        return tracts
    
    @classmethod
    def visualizeTracts(cls):
        from descartes import PolygonPatch
        f = plt.figure(figsize=(6,6))
        ax = f.gca()
        for k, t in cls.tracts.items():
            ax.add_patch(PolygonPatch(t.polygon, alpha=0.5, fc="green"))
        ax.axis("scaled")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig("tracts.png")



def compare_tract_shapefiles():
    """There are two version of tract level shapfiles.
    Are they the same?"""
    trts1 = Tract.createAllTractObjects()
    trts2 = Tract.createAllTractObjects("data/chicago-shp-2010-gps/chicago_tract_wgs84")
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
#    t1, t2 = compare_tract_shapefiles()
    trts1 = Tract.createAllTractObjects()
    Tract.visualizeTracts()
