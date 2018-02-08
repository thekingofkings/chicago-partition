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
                           comm_to_plot = [47,49,50])

CommunityArea.visualizeCAs(fname='{}/before-{}.pdf'.format(project_name,targetName),
                           plot_measure=targetInit,
                           labels=True,
                           title='',
                           case_study=True,
                           comm_to_plot = [47,49,50])



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
                           jitter_labels=True)



CommunityArea.visualizeCAs(fname='{}/after-{}.pdf'.format(project_name,targetName),
                           plot_measure=targetFinal,
                           labels=True,
                           title='',
                           case_study=True,
                           comm_to_plot=[47, 49, 50],
                           jitter_labels=True)





featureBefore = singleFeatureForStudyInit.ix[cas]
targetBefore = targetInit.ix[cas]

featureAfter = singleFeatureForStudyFinal.ix[cas]
targetAfter = targetFinal.ix[cas]

corr_before = np.corrcoef(x=featureBefore,y=targetBefore)
corr_after = np.corrcoef(x=featureAfter,y=targetAfter)


print "Correlation Before: "
print corr_before
print "Correlation After: {}"
print corr_after



######## House Price Prediction ########

import MCMC
from community_area import CommunityArea
from tract import Tract
import numpy as np

# Global variables
project_name = 'case-study-house-price'
targetName = 'train_average_house_price'
singleFeatureName = 'poverty_index'
finalPartitionFile = 'q-learning-v10-final-partition.txt'

cas = [23,26]


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
                           comm_to_plot = cas)

CommunityArea.visualizeCAs(fname='{}/before-{}.pdf'.format(project_name,targetName),
                           plot_measure=targetInit,
                           labels=True,
                           title='',
                           case_study=True,
                           comm_to_plot = cas)

for ca_i in cas:
    example_ca = CommunityArea.CAs[ca_i]
    print ca_i
    for t_id in example_ca.tracts.keys():
        t = example_ca.tracts[t_id]
        centroid = (t.centroid[1],t.centroid[0])
        print t_id, centroid

    Tract.visualizeTracts(tractIDs=example_ca.tracts.keys(),fname='plots/{}/before-tracts-{}.pdf'.format(project_name,ca_i),labels=True)




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
                           jitter_labels=True)



CommunityArea.visualizeCAs(fname='{}/after-{}.pdf'.format(project_name,targetName),
                           plot_measure=targetFinal,
                           labels=True,
                           title='',
                           case_study=True,
                           comm_to_plot=cas,
                           jitter_labels=True)





featureBefore = singleFeatureForStudyInit.ix[cas]
targetBefore = targetInit.ix[cas]

featureAfter = singleFeatureForStudyFinal.ix[cas]
targetAfter = targetFinal.ix[cas]

corr_before = np.corrcoef(x=featureBefore,y=targetBefore)
corr_after = np.corrcoef(x=featureAfter,y=targetAfter)


print "Correlation Before: "
print corr_before
print "Correlation After: {}"
print corr_after







