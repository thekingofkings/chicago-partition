import MCMC
import geopandas as gp
import matplotlib.pyplot as plt
from community_area import CommunityArea
from tract import Tract

# Global variables
project_name = 'case-study-crime'
targetName = 'total'
singleFeatureName = 'poverty_index'
finalPartitionFile = 'q-learning-v10-final-partition.txt'


# Initialize MCMC: learn regression using administrative boundaries
MCMC.initialize(project_name=project_name, targetName=targetName, lmbd=0.75, f_sd=10, Tt=10)

# Collect x,y features from community areas
singleFeatureForStudyInit = CommunityArea.features[singleFeatureName].copy()
targetInit = CommunityArea.features[targetName].copy()

# Plot x,y by community area
CommunityArea.visualizeCAs(fname='{}/before-{}.png'.format(project_name,singleFeatureName),
                           plot_measure=singleFeatureForStudyInit,
                           labels=True,
                           title = 'Before: {}'.format(singleFeatureName))

CommunityArea.visualizeCAs(fname='{}/before-{}.png'.format(project_name,targetName),
                           plot_measure=targetInit,
                           labels=True,
                           title='Before: Crime count')


# Read in optimal partition
Tract.readPartition(finalPartitionFile)
# Update features conditional on new community partition
CommunityArea.createAllCAs(Tract.tracts)

# Collect x,y features from community areas
singleFeatureForStudyFinal = CommunityArea.features[singleFeatureName].copy()
targetFinal = CommunityArea.features[targetName].copy()
# Visualize x,y
CommunityArea.visualizeCAs(fname='{}/after-{}.png'.format(project_name,singleFeatureName),
                           plot_measure=singleFeatureForStudyFinal,
                           labels=True,
                           title = "After: {}".format(singleFeatureName))



CommunityArea.visualizeCAs(fname='{}/after-{}.png'.format(project_name,targetName),
                           plot_measure=targetFinal,
                           labels=True,
                           title = "After: Crime Count")
