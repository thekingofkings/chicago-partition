import MCMC
from community_area import CommunityArea
from tract import Tract
import numpy as np

####### Crime Prediction ########
# Global variables
project_name = 'case-study-crime'
targetName = 'total'
singleFeatureName = 'poverty_index'
finalPartitionFile = 'q-learning-v10-final-partition.txt'
arrow = (-87.6395, 41.661, 0.02, 0.02)

cas = [47,49,50]


# Initialize MCMC: learn regression using administrative boundaries
MCMC.initialize(project_name=project_name, targetName=targetName, lmbd=0.75, f_sd=10, Tt=10)

# Collect x,y features from community areas
singleFeatureForStudyInit = CommunityArea.features[singleFeatureName].copy()
targetInit = CommunityArea.features[targetName].copy()

# Plot x,y by community area
CommunityArea.visualizeCAs(fname='{}/before-{}.pdf'.format(project_name,singleFeatureName),
                           plot_measure=singleFeatureForStudyInit,
                           labels=True,
                           title='',
                           case_study=True,
                           comm_to_plot = [47,49,50],
                           arrow=arrow)

CommunityArea.visualizeCAs(fname='{}/before-{}.pdf'.format(project_name,targetName),
                           plot_measure=targetInit,
                           labels=True,
                           title='',
                           case_study=True,
                           comm_to_plot = [47,49,50],
                           arrow=arrow)



# Read in optimal partition
Tract.readPartition(finalPartitionFile)
# Update features conditional on new community partition
CommunityArea.createAllCAs(Tract.tracts)

# Collect x,y features from community areas
singleFeatureForStudyFinal = CommunityArea.features[singleFeatureName].copy()
targetFinal = CommunityArea.features[targetName].copy()
# Visualize x,y
CommunityArea.visualizeCAs(fname='{}/after-{}.pdf'.format(project_name,singleFeatureName),
                           plot_measure=singleFeatureForStudyFinal,
                           labels=True,
                           title='',
                           case_study=True,
                           comm_to_plot=[47, 49, 50],
                           jitter_labels=True,
                           arrow=arrow)



CommunityArea.visualizeCAs(fname='{}/after-{}.pdf'.format(project_name,targetName),
                           plot_measure=targetFinal,
                           labels=True,
                           title='',
                           case_study=True,
                           comm_to_plot=[47, 49, 50],
                           jitter_labels=True,
                           arrow=arrow)





## Plot again for paper without heat map

CommunityArea.visualizeCAs(fname='q-learning-v10-CAs.pdf',
                           labels=False,
                           title='')



######## House Price Prediction ########

import MCMC
from community_area import CommunityArea
from tract import Tract
import numpy as np

# Global variables
project_name = 'case-study-house-price'
targetName = 'train_average_house_price'
singleFeatureName = 'poverty_index'
finalPartitionFile = 'q-learning-v1-final-partition.txt'

ex_house = None


cas = [3,5,6,7]


# Initialize MCMC: learn regression using administrative boundaries
MCMC.initialize(project_name=project_name, targetName=targetName, lmbd=0.75, f_sd=10, Tt=10)

# Collect x,y features from community areas
singleFeatureForStudyInit = CommunityArea.features[singleFeatureName].copy()
targetInit = CommunityArea.features[targetName].copy()

# Plot x,y by community area



CommunityArea.visualizeCAs(fname='{}/before-{}.pdf'.format(project_name,singleFeatureName),
                           plot_measure=singleFeatureForStudyInit,
                           labels=True,
                           title='',
                           case_study=True,
                           comm_to_plot = cas,
                           arrow=None)

CommunityArea.visualizeCAs(fname='{}/before-{}.pdf'.format(project_name,targetName),
                           plot_measure=targetInit,
                           labels=True,
                           title='',
                           case_study=True,
                           comm_to_plot = cas,
                           arrow=None)

CommunityArea.visualizeCAs(fname='{}/before-{}-all.pdf'.format(project_name,targetName),
                           plot_measure=targetInit,
                           labels=False,
                           title='',
                           case_study=False,
                           comm_to_plot = None,
                           arrow=None)




# Read in optimal partition
Tract.readPartition(finalPartitionFile)
# Update features conditional on new community partition
CommunityArea.createAllCAs(Tract.tracts)

# Collect x,y features from community areas
singleFeatureForStudyFinal = CommunityArea.features[singleFeatureName].copy()
targetFinal = CommunityArea.features[targetName].copy()
# Visualize x,y
CommunityArea.visualizeCAs(fname='{}/after-{}.pdf'.format(project_name,singleFeatureName),
                           plot_measure=singleFeatureForStudyFinal,
                           labels=True,
                           title='',
                           case_study=True,
                           comm_to_plot=cas,
                           jitter_labels=False,
                           arrow=ex_house)



CommunityArea.visualizeCAs(fname='{}/after-{}.pdf'.format(project_name,targetName),
                           plot_measure=targetFinal,
                           labels=True,
                           title='',
                           case_study=True,
                           comm_to_plot=cas,
                           jitter_labels=False,
                           arrow=ex_house)
racts(tractIDs=tracts,fname='{}/tracts-{}-after-{}.pdf'.format(c,project_name,targetName))