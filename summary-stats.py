from tract import Tract
from community_area import CommunityArea
import pandas as pd
import numpy as np

def getSummaryStats(years):
    means = []
    for y in years:

        Tract.createAllTracts()
        CommunityArea.createAllCAs(Tract.tracts)

        CommunityArea._initializeCAfeatures(crimeYear=y)
        feature_list = CommunityArea.featureNames + ['total','train_average_house_price','test_average_house_price']
        X = CommunityArea.features[feature_list]
        mean_y = X.mean()
        means.append(mean_y)


    summary_stats = pd.concat(means,axis=1)
    summary_stats.columns = years


    return np.round(summary_stats.transpose(),2)



if __name__ == '__main__':
    years = [2010, 2011]
    stats = getSummaryStats(years)
    stats.to_csv("output/summary-stats.csv")
    print stats
