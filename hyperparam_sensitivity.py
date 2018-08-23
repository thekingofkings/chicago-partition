from tract import Tract
from community_area import CommunityArea

max_m = 77
plot = True

m_grid = range(0,max_m)
m_grid = sorted(m_grid, reverse=True)

for i in m_grid:
    print "Randomly combining partitions into {} regions...".format(i)
    # Estimate models here
    if i == (max_m - 1):
        Tract.createAllTracts()
        Tract.generateFeatures(2011)
        CommunityArea.createAllCAs(Tract.tracts)

    CommunityArea.rand_init_communities(i)

    if plot:
        CommunityArea.visualizeCAs(fname='random-community-init-{}.png'.format(i), labels=True, iter_cnt=i)