#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:14:08 2018

@author: hxw186

# Calculate the average house price.

This file should be only run once to generate the house price features.


Input:
    data/house_source_extra.csv
Output:
    data/house_price_features
The output file should be used in `Tract.generateFeatures` to insert house
price features into dataframe.


Key points:
    1. The house prices are split temporally into *old* and *new*, which are 
    used for training and testing respectively.
    2. The temporal splition should be roughly 1:1, so that we should not worry
    about data sparseness
"""

from tract import Tract
from shapely.geometry import Point
import pandas as pd
import pickle


def get_house_price():
    """
    Return a dataframe of houses. Each row is one house.
    Preprocess includes:
        1. filter NaN value
        2. filter outlier house prices
    """
    houses = pd.read_csv("data/house_source_extra.csv")
    houses = houses[["soldTime", "priceSqft", "lat", "lon"]]
    houses = houses[(houses["priceSqft"] > 30) & (houses["priceSqft"] < 3000)]
    return houses


def split_house_data(houses):
    """
    Split house price data into train and test.
    The split condition is the sold date.
    The training test ratio is roughly 1:1
    """
    houses["soldDate"] = houses["soldTime"].apply(lambda x: 100 * int(x[0:2])
                        + int(x[3:5]) + 10000 * int(x[6:]))
    median_sold_date = houses["soldDate"].median()
    print "Split point is {}".format(int(median_sold_date))
    train_houses = houses[houses["soldDate"] <= median_sold_date]
    test_houses = houses[houses["soldDate"] > median_sold_date]
    return train_houses, test_houses



if __name__ == '__main__':
    houses = get_house_price()
    train, test = split_house_data(houses)
    