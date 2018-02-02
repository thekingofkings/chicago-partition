import MCMC
from community_area import CommunityArea
from tract import Tract
import numpy as np

# Global varialces
project_name = 'case-study-crime'
targetName = 'total'


MCMC.initialize(project_name=project_name, targetName=targetName, lmbd=0.75, f_sd=1.5, Tt=10)
# Error = y_true - y_hat

argmax_error = np.argmax(MCMC.errors1.values)
ca_main = CommunityArea.CAs[argmax_error]
feature_names = ['B1901Z01',
                 'B1901H01',
                 'B1901B01',
                 'B1901I01',
                 'B1901D01']


tract_ids = []
for t in ca_main.tracts.values():
    tract_ids.append(t.id)




X_all_tract = Tract.features


X_ca_tracts = X_all_tract.ix[tract_ids]




print "Mean tract-crime - all tracts: ", X_all_tract.total.mean()
print "Mean tract-crime - community of interest: ", X_ca_tracts.total.mean()


features = X_all_tract.columns

for x_i in features:
    x_i_mean_all = np.mean(X_all_tract[x_i])
    x_i_mean_ca_tracts = np.mean(X_ca_tracts[x_i])
    pct_diff = (x_i_mean_ca_tracts - x_i_mean_all) / x_i_mean_all

    print "----{}----".format(x_i)
    print "Mean tract-level {:s} - all tracts: {:.4f} ".format(x_i,x_i_mean_all)
    print "Mean tract-level {:s} - community of interest: {:.4f}".format(x_i,x_i_mean_ca_tracts)
    print "Percentage change: {:.4f}".format(pct_diff)
    print ""

#X_ca_tracts.to_csv("output/case-study-features.csv")

#CommunityArea.visualizeCAs(iter_cnt=None,fname=project_name+"-CAs-iter-final.png")


