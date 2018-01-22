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
    data/house_price_features.df
The output file should be used in `Tract.generateFeatures` to insert house
price features into dataframe.

Key points:
    1. The house prices are split temporally into *old* and *new*, which are 
    used for training and testing respectively.
    2. The temporal splition should be roughly 1:1, so that we should not worry
    about data sparseness

Use `e = pd.DataFrame.from_csv("data/house_price_features.df")` to retrieve
generated price features.
"""

from tract import Tract
from shapely.geometry import Point
import pandas as pd


def get_house_price():
    """
    Return a dataframe of houses. Each row is one house.
    Preprocess includes:
        1. filter NaN value
        2. filter outlier house prices
    """
    houses = pd.read_csv("data/house_source_extra.csv")
    houses = houses[["soldTime", "priceSqft", "lat", "lon"]]
    houses = houses[(houses["priceSqft"] > 50) & (houses["priceSqft"] < 3000)]
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


def calculate_tract_house_price(houseDF, tracts):
    """
    Calculate the sum price_per_sqrt and the number of estate within a tract.
    Input:
        houseDF - a dataframe of house price and coordinate.
        tracts - all tracts with their boundary
    """
    house_cnt = {}
    sum_unit_price = {}
    for idx, house in houseDF.iterrows():
        coord = Point(house.lon, house.lat)
        for k, t in tracts.items():
            if t.polygon.contains(coord):
                if k not in house_cnt:
                    house_cnt[k] = 1
                    sum_unit_price[k] = house.priceSqft
                else:
                    house_cnt[k] += 1
                    sum_unit_price[k] += house.priceSqft
                break
    return house_cnt, sum_unit_price


def save_house_price_features():
    houses = get_house_price()
    train, test = split_house_data(houses)
    tracts = Tract.createAllTracts(calculateAdjacency=False)
    train_cnt, train_price = calculate_tract_house_price(train, tracts)
    test_cnt, test_price = calculate_tract_house_price(test, tracts)
    tractIDs = tracts.keys()
    price_features_dict = {'train_count': [], 'train_price': [], 
                           'test_count': [], 'test_price': []}
    for k in tractIDs:
        if k in train_cnt:
            price_features_dict['train_count'].append(train_cnt[k])
            price_features_dict['train_price'].append(train_price[k])
        else:
            price_features_dict['train_count'].append(0)
            price_features_dict['train_price'].append(0)
        if k in test_cnt:
            price_features_dict['test_count'].append(test_cnt[k])
            price_features_dict['test_price'].append(test_price[k])
        else:
            price_features_dict['test_count'].append(0)
            price_features_dict['test_price'].append(0)

    price_features = pd.DataFrame(price_features_dict, index=tractIDs)
    price_features.to_csv("data/house_price_features.df")
    return price_features


if __name__ == '__main__':
    r = save_house_price_features()


