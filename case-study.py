import MCMC
import os
import geopandas as gp
from community_area import CommunityArea
from feature_utils import retrieve_income_features
from tract import Tract
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from pandas import DataFrame

# Global vars
project_name = 'case-study-crime'
targetName = 'total'


# Functions



def getIncomeFeatureNames():
    features = []
    ethnics = ['H','B','I','D']
    # Racial features
    for ethnic in ethnics:
        feature_col_name = 'B1901%s01'%ethnic
        features.append(feature_col_name)

    # Income by race features
    for ethnic in ethnics:
        weight_features = ['B1901%s%s'%(ethnic, str(level).zfill(2)) for level in range(2,18)]
        features += weight_features
    print "Number of features %s " % len(features)
    return features

def getGeoData(parentDirectory=None):
    # Load shape file
    if parentDirectory is None:
        parentDirectory = ''

    chicago_tract_geod = gp.GeoDataFrame.from_file(
        parentDirectory + 'data/Census-Tracts-2010/chicago-tract.shp',
        encoding='utf-8'
    )
    chicago_ca_geod = gp.GeoDataFrame.from_file(
        parentDirectory + 'data/Cencus-CA-Now/chicago-ca-file.shp',
        encoding='utf-8'
    )
    chicago_tract_geod.columns = ['statefp10', 'countyfp10', 'tractID', 'namelsad10', 'communityID', 'geoid10',
                                  'commarea_n', 'name10', 'notes', 'geometry']
    chicago_tract_geod['tractID'] = chicago_tract_geod['tractID'].astype(int)

    chicago_ca_geod.columns = ['perimeter', 'community', 'shape_len', 'shape_area', 'area', 'comarea', 'area_numbe',
                               'communityID', 'comarea_id', 'geometry']

    # Change type of community to int
    chicago_ca_geod['communityID'] = chicago_ca_geod['communityID'].apply(int)
    chicago_tract_geod['communityID'] = chicago_tract_geod['communityID'].apply(int)

    # Change type of tract ids to int
    chicago_tract_geod['tractID'] = chicago_tract_geod['tractID'].apply(int)

    return chicago_tract_geod, chicago_ca_geod

parentDirectory = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "/"
tractGeoData, caGeoData = getGeoData(parentDirectory)


MCMC.initialize(project_name=project_name, targetName=targetName, lmbd=0.75, f_sd=1.5, Tt=10)
# Error = y_true - y_hat

errors = MCMC.errors1.to_frame('error')


caGeoData = caGeoData.merge(errors, left_on ='communityID',right_index = True)
argmax_error = np.argmax(MCMC.errors1.values)
# Plot results, label with community ID numbers

fig, ax = plt.subplots()
caGeoData.plot(ax=ax, edgecolor='red', alpha=0.8, linewidth=0.2, column='error', cmap='OrRd',figsize=(12,8))

for i, row in caGeoData.iterrows():
    ax.text(row.geometry.centroid.x,
            row.geometry.centroid.y,
            int(row.communityID),
            horizontalalignment='center',
            verticalalignment='center')
plt.title("Errors")
plt.savefig("plots/{}/community-area-map.png".format(project_name))
plt.close()

#CommunityArea.visualizeCAs(iter_cnt='initial',fname="{}/community-area-map.png".format(project_name))

ca_main = CommunityArea.CAs[argmax_error]
feature_names = getIncomeFeatureNames()
# add total crime figures
feature_names += ["total"]

_, feature_dict = retrieve_income_features()



# Write Interpretable feature names to file

f = open('output/{}-feature-names.txt'.format(project_name),'w')
for key in feature_dict.keys():
    f.write("{}: {} \n".format(key,feature_dict[key]))

f.close()

tract_ids = []
for t in ca_main.tracts.values():
    tract_ids.append(t.id)


#tmp = Tract.features
X_all_tract = Tract.features[feature_names]
X_ca_tracts = X_all_tract.ix[tract_ids][feature_names]

X_ca_tracts.rename(feature_dict,axis = 'columns',inplace=True)

for col in X_ca_tracts.columns:
    X_ca_tracts[col].plot(kind='bar',figsize=(16,12))
    plt.savefig("plots/{}/{}.png".format(project_name,col))
    plt.close()
    plt.clf()




for x_i in feature_names:
    if x_i == 'total':
        good_feature_name = 'total'
    else:
        good_feature_name = feature_dict[x_i]
    x_i_mean_all = np.mean(X_all_tract[x_i])
    x_i_mean_ca_tracts = np.mean(X_ca_tracts[good_feature_name])
    pct_diff = (x_i_mean_ca_tracts - x_i_mean_all) / x_i_mean_all



    print "----{}----".format(good_feature_name)
    print "Mean tract-level {:s} - all tracts: {:.4f} ".format(x_i,x_i_mean_all)
    print "Mean tract-level {:s} - community of interest: {:.4f}".format(x_i,x_i_mean_ca_tracts)
    print "Percentage change: {:.4f}".format(pct_diff)
    print ""


dist_mtx = pairwise_distances(X_ca_tracts.values)
#dist_mtx = (dist_mtx - np.mean(dist_mtx.flatten())) / np.std(dist_mtx.flatten())


fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(dist_mtx,cmap='bwr',interpolation='nearest')
ax.set_xlabel("Tract ID",fontsize = 18)
ax.set_ylabel("Tract ID",fontsize = 18)
for tick in ax.get_xticklabels():
    tick.set_fontsize(16)
for tick in ax.get_yticklabels():
    tick.set_fontsize(16)
plt.savefig("plots/{}/heat-map.png".format(project_name))
plt.close()
plt.clf()