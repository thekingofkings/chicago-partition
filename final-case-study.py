import MCMC
import geopandas as gp
import matplotlib.pyplot as plt
from community_area import CommunityArea
from tract import Tract

# Global variables
project_name = 'case-study-crime'
targetName = 'total'
singleFeatureName = 'income_variance'
finalPartitionFile = 'q-learning-v10-final-partition.txt'


# Initialize MCMC: learn regression using administrative boundaries
MCMC.initialize(project_name=project_name, targetName=targetName, lmbd=0.75, f_sd=10, Tt=10)

singleFeatureForStudyInit = CommunityArea.singleFeature.copy()
CommunityArea.visualizeCAs(fname='{}/before-{}.png'.format(project_name,singleFeatureName),
                           by=singleFeatureForStudyInit,
                           labels=True,
                           title = 'Before: {}'.format(singleFeatureName))

targetInit = CommunityArea.features[targetName].copy()
CommunityArea.visualizeCAs(fname='{}/before-{}.png'.format(project_name,targetName),
                           by=targetInit,
                           labels=True,
                           title='Before: Crime count')



Tract.readPartition(finalPartitionFile)
CommunityArea.createAllCAs(Tract.tracts,singleFeature='income_variance')


singleFeatureForStudyFinal = CommunityArea.singleFeature.copy()
CommunityArea.visualizeCAs(fname='{}/after-{}.png'.format(project_name,singleFeatureName),
                           by=singleFeatureForStudyFinal,
                           labels=True,
                           title = "After: {}".format(singleFeatureName))


targetFinal = CommunityArea.features[targetName].copy()
CommunityArea.visualizeCAs(fname='{}/after-{}.png'.format(project_name,targetName),
                           by=targetFinal,
                           labels=True,
                           title = "After: Crime Count")
