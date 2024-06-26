"""'Helper functions for auto car prediction."""
import logging
import pickle
import sys
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd

import config
from exception import CustomException


def log_price_info(test_data_frame: pd.DataFrame, price_info: Dict[str, Any]) \
        -> None:
    """Log model type and corresponding predicted car prices."""
    model_lst = price_info['Model']
    price_lst = price_info['Predicts']

    logging.info("Current test car data frame and parameters:")
    logging.info("\n %s", test_data_frame.to_string())

    for model, price in zip(model_lst, price_lst):
        logging.info("Model type: %s", model)
        logging.info("Predicted car price: %f", price)


def replace_missing(data_frame: pd.DataFrame) \
        -> pd.DataFrame:
    """Replace missed values from the loaded file with 'NA'."""
    data_frame.replace('?', np.nan, inplace=True)

    return data_frame


def mean_imputation(data_frame: pd.DataFrame,
                    num_attrs: Tuple) \
        -> pd.DataFrame:
    """Replace NA value in numeric attribute with mean attr value."""
    for attr in num_attrs:
        mean_cat = data_frame[attr].mean()
        data_frame[attr].fillna(mean_cat, inplace=True)

    return data_frame


def mode_imputation(data_frame: pd.DataFrame,
                    cat_attrs: Tuple) \
        -> pd.DataFrame:
    """Replace missing category value with the most common attr value."""
    for attr in cat_attrs:
        mode_cat = data_frame[attr].mode()[0]
        data_frame[attr].fillna(mode_cat, inplace=True)

    return data_frame


def zero_imputation(data_frame: pd.DataFrame,
                    cat_ord_attrs: Tuple) \
        -> pd.DataFrame:
    """Replace the missing category ordinal value with 0."""
    for attr in cat_ord_attrs:
        data_frame[attr].fillna(0, inplace=True)

    return data_frame


def convert_cat_ord_to_num(data_frame: pd.DataFrame) \
        -> pd.DataFrame:
    """Convert cat ordinal to numerical in order to compute correlation."""
    data_frame['normalized-losses'] = data_frame['normalized-losses'].dropna(
                                        ).astype(int)
    data_frame['bore'] = data_frame['bore'].dropna().astype(float)
    data_frame['stroke'] = data_frame['stroke'].dropna().astype(float)
    data_frame['horsepower'] = data_frame['horsepower'].dropna().astype(int)
    data_frame['peak-rpm'] = data_frame['peak-rpm'].dropna().astype(int)
    data_frame['price'] = data_frame['price'].dropna().astype(int)

    return data_frame


def get_unique_cat_values(data_frame: pd.DataFrame,
                          cat_attrs: Tuple) \
        -> Dict[str, Tuple[str]]:
    """Get unique values from the categorical attributes of auto dataset."""
    unq_cat_values = {}
    for cat in cat_attrs:
        unq_cat_values[cat] = tuple(data_frame[cat].unique())

    logging.info("Extract unique values for categorical attributes.")

    return unq_cat_values


def save_model(model: Any) \
        -> None:
    """Save a car price predicted model."""
    try:
        filename = config.PATH['modelpathname'] \
            / config.FILE['modelfilename']
        with open(filename, 'wb') as file:
            pickle.dump(model, file)

    except Exception as err:
        raise CustomException(err, sys) from None

    logging.info("Save price prediction model as a file.")


def load_model() -> Any:
    """Load a car price predicted model."""
    try:
        filename = config.PATH['modelpathname'] \
            / config.FILE['modelfilename']
        with open(filename, 'wb') as file:
            model = pickle.load(file)

    except Exception as err:
        raise CustomException(err, sys) from None

    logging.info("Load price prediction model.")

    return model
