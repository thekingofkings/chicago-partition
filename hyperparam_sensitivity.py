from tract import Tract
from community_area import CommunityArea
from MCMC import naive_MCMC, MCMC_softmax_proposal

max_m = 77
min_m = 50
plot = True

m_grid = range(min_m,max_m+1)
m_grid = sorted(m_grid, reverse=True)

for i in m_grid:

    # Estimate models here
    if i == (max_m):
        print "Initializing {} regions".format(max_m)
        Tract.createAllTracts()
        Tract.generateFeatures(2011)
        CommunityArea.createAllCAs(Tract.tracts)
    else:
        print "Randomly combining {} regions into {} regions...".format(i + 1, i)
        CommunityArea.rand_init_communities(i)
        print "Dimensions of updated design matrix: {}".format(CommunityArea.features.shape)


    if plot:
        CommunityArea.visualizeCAs(fname='random-community-init-{}.png'.format(i), labels=True, iter_cnt=i)

    # estimate naive MCMC
    naive_MCMC('crime-naive-sensitivity-{}'.format(i), targetName='total',
               lmbda=0.005, f_sd=5, Tt=0.1, init_ca=False)
    # estimate MCMC with softmax proposal
    MCMC_softmax_proposal('crime-softmax-sensitivity-{}'.format(i), targetName='total',
                          lmbda=0.005, f_sd=5, Tt=0.1, init_ca=False)