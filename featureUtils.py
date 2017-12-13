#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 21:02:07 2017

@author: kok

Generate tract-level features for regressions
"""


from openpyxl import load_workbook
import pandas as pd


def retrieve_corina_features():
    from pandas import read_stata
    
    r = read_stata('data/SE2000_AG20140401_MSAcmsaID.dta')
    cnt = 0
    header = ['pop00', 'ppov00', 'disadv00', 'pdensmi00', 'hetero00', 'phisp00', 'pnhblk00']
    
    fields_dsp = ['total population', 'poverty index', 'disadvantage index', 
                  'population density', 'ethnic diversity', 'pct hispanic', 'pct black']

    ST = {}
    for row in r.iterrows():
        tract = row[1]
        if tract['statetrim'] == '17' and tract['countrim'] == '031':
            cnt += 1
            tl = []
            tid = '17031' + tract['tracttrim']
            for h in header:
                tl.append(tract[h])
            ST[tid] = tl
    return fields_dsp, ST


def retrieve_income_features():
    """
    read the xls file '../data/Household Income by Race and Census Tract and Community Area.xlsx'
    """
    wb = load_workbook('data/Household Income by Race and Census Tract and Community Area.xlsx')
    ws = wb.active
    tractColumn = [cell.value for cell in ws['h']]
    dataColumns = ws['K1:DU890']
    
    header = []
    header_description = []
    income_features = []
    tractIDs = []
    for idx, tractID in enumerate(tractColumn):
        if idx == 0:
            header = [cell.value for cell in dataColumns[idx]]
        elif idx == 1:
            header_description = [cell.value for cell in dataColumns[idx]]
        elif idx == 2:
            header_description = ["{} {}".format(header_description[i], cell.value)
                                  for i, cell in enumerate(dataColumns[idx])]
        else:
            if tractID != None:
                tractIDs.append(int("17031"+tractID))
                row = [cell.value for cell in dataColumns[idx]]
                income_features.append(row)
    featureDF = pd.DataFrame(data=income_features, index=tractIDs, columns=header)
    header_decode = dict(zip(header, header_description))
    return featureDF, header_decode


def retrieve_crime_count(year=2010):
    """
    Output:
        Dataframe of crime counts in various categoires. The index type is integer.
        The column name is crime type
    """
    Y = pd.read_csv("data/chicago-crime-tract-level-{0}.csv".format(year),
                    header=0, index_col=0)
    return Y
        


def validate_region_keys():
    """
    We have three sources of tract IDs: 
        1) shapefile probably from 2010
        2) the census demographics from 2000 (Corina provides)
        3) the census demographics from 2010 (download from web)
    
    The question to figure out is which year the shapefile is from? - confirmed from 2010.
    Further, are there any missing tracts in the shapefile? - yes. there are 8
    """
    from tract import Tract
    tracts = Tract.createAllTracts()
    shp_keys = set(tracts.keys())
    
    fields_dsp, st = retrieve_corina_features() # 2000 census data
    corina_keys = set(st.keys())
    
    f, d = retrieve_income_features()  # 2010 census data
    census2010_keys = set(f.index)
    
    print "Compare the tract IDs between shapefile and 2000 census"
    print "len(shp) {}, len(2000 census) {}".format(len(shp_keys), len(corina_keys))
    print "intersection: {}, union {}.".format(len(shp_keys & corina_keys), 
                         len(shp_keys | corina_keys))
    
    print "Compare the tract IDs between shapefile and 2010 census"
    print "len(shp) {}, len(2010 census) {}".format(len(shp_keys), len(census2010_keys))
    print "intersection: {}, union {}.".format(len(shp_keys & census2010_keys), 
                         len(shp_keys | census2010_keys))
    
    for s in census2010_keys:
        if s not in shp_keys:
            print s

if __name__ == '__main__':
    validate_region_keys()
    f,d = retrieve_income_features()
    y = retrieve_crime_count()