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
    """
    Get a list of all tract-level features that are used in Hongjian's feature aggregation routine.
    Mainly consists of race and income degraphics
    :return: List of features
    """
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
    """
    Query census data and return as GeoDataFrame for easier plotting
    :param parentDirectory: Project parent directory (str)
    :return: [0]: GeoDataFrame of all tract-level data
            [1]: GeoDataFrame of community-level data
    """
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


# Save parent directory (access data folder above)
parentDirectory = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "/"
# Query chicago geo data
tractGeoData, caGeoData = getGeoData(parentDirectory)

# Initialize MCMC: learn regression using administrative boundaries
MCMC.initialize(project_name=project_name, targetName=targetName, lmbd=0.75, f_sd=1.5, Tt=10)


# Regression errors: error = y_true - y_hat
errors = MCMC.errors1.to_frame('error')

# Merge errors to full GeoDataFrame
caGeoData = caGeoData.merge(errors, left_on ='communityID',right_index = True)
# Find community with largest error
argmax_error = np.argmax(MCMC.errors1.values)
# Commnity object with largest error
ca_main = CommunityArea.CAs[argmax_error]

# Get list of tract-level features
feature_names = getIncomeFeatureNames()
# add total crime figures
feature_names += ["total"]

# Get a dictionary of census feature names (keys), and human readable feature names (values)
_, feature_dict = retrieve_income_features()

# Write Interpretable feature names to file
f = open('output/{}-feature-names.txt'.format(project_name),'w')
for key in feature_dict.keys():
    f.write("{}: {} \n".format(key,feature_dict[key]))
f.close()

tract_ids = []
for t in ca_main.tracts.values():
    tract_ids.append(t.id)

X_all_tract = Tract.features[feature_names]
X_ca_tracts = X_all_tract.ix[tract_ids][feature_names]
X_ca_tracts.rename(feature_dict,axis = 'columns',inplace=True)


# Print deviance of given feature x_i from global mean
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

# Compute distance matrix of all tracts
dist_mtx = pairwise_distances(X_ca_tracts.values)

# Plot error map and distance matrix
fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(18,8))
caGeoData.plot(ax=ax[0], edgecolor='black', alpha=0.8, linewidth=0.2, column='error', cmap='OrRd')
for i, row in caGeoData.iterrows():
    ax[0].text(row.geometry.centroid.x,
            row.geometry.centroid.y,
            int(row.communityID),
            horizontalalignment='center',
            verticalalignment='center',fontsize=12)
ax[0].set_title("(a) Regression errors by administrative boundary",fontsize=20)

ax[1].imshow(dist_mtx,cmap='bwr',interpolation='nearest')
ax[1].set_xlabel("Tract ID",fontsize = 18)
ax[1].set_ylabel("Tract ID",fontsize = 18)
ax[1].set_title("(b) Similarity matrix of tracts within community #6",fontsize=20)

plt.savefig("plots/{}/intro-case-study.png".format(project_name))
plt.close()
plt.clf()