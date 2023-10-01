"""'Helper functions for auto car prediction."""
from typing import Tuple

import numpy as np
import pandas as pd


def replace_missing(data_frame: pd.DataFrame):
    """Replaced missed values from the loaded file with 'NA.'"""
    data_frame.replace('?', np.nan, inplace = True)
    return data_frame

def mean_imputation(data_frame: pd.DataFrame, num_attrs: Tuple):
    """Replace NA value in numeric attribute with mean attr value."""
    for attr in num_attrs:
        mean_cat = data_frame[attr].mean()
        data_frame[attr].fillna(mean_cat,inplace=True)
    return data_frame

def mode_imputation(data_frame, cat_attrs: Tuple):
    """Replace missing category value with the most common attr value."""
    for attr in cat_attrs:
        mode_cat = data_frame[attr].mode()[0]
        data_frame[attr].fillna(mode_cat, inplace=True)
    return data_frame

def zero_imputation(data_frame: pd.DataFrame, cat_ord_attrs: Tuple):
    """Replace the missing category ordinal value with 0."""
    for attr in cat_ord_attrs:
        data_frame[attr].fillna(0, inplace=True)
    return data_frame

def convert_cat_ord_to_num(data_frame: pd.DataFrame):
    """Convert cat ordinal to numerical in order to compute correlation."""
    data_frame['normalized-losses'] = data_frame['normalized-losses'].dropna().astype(int)
    data_frame['bore'] = data_frame['bore'].dropna().astype(float)
    data_frame['stroke'] = data_frame['stroke'].dropna().astype(float)
    data_frame['horsepower'] = data_frame['horsepower'].dropna().astype(int)
    data_frame['peak-rpm'] = data_frame['peak-rpm'].dropna().astype(int)
    data_frame['price'] = data_frame['price'].dropna().astype(int)

    return data_frame
