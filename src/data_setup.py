"""Load, save and generate data operations."""
import logging
import random
import sys
from typing import Dict, Tuple

import pandas as pd

import config
from exception import CustomException


def load_data() \
        -> pd.DataFrame:
    """Load raw auto data."""
    logging.info("Enter train auto data")
    try:
        filename = config.PATH['datapathname'] / config.FILE['datafilename']
        data_frame = pd.read_csv(filename)
        logging.info("Read auto dataset as a dataframe.")

    except Exception as err:
        raise CustomException(err, sys) from None

    return data_frame


def random_generate_test_car(cat_uniq_dict: Dict[str, Tuple[str]]) \
        -> pd.DataFrame:
    """Generate a test car with random parameter values met in auto dataset."""
    rand_car_props = {}
    for key, val in cat_uniq_dict.items():
        rand_car_props[key] = [random.choice(list(val))]

    test_data_frame = pd.DataFrame.from_dict(rand_car_props)

    return test_data_frame


def generate_test_car() \
        -> pd.DataFrame:
    """Select scar parameters for test purpose."""
    test_car = {'make': ['audi'],
                'fuel-type': ['gas'],
                'aspiration': ['std'],
                'num-of-doors': ['four'],
                'body-style': ['sedan'],
                'drive-wheels': ['4wd'],
                'engine-location': ['front'],
                'engine-type': ['ohc'],
                'num-of-cylinders': ['five'],
                'fuel-system': ['2bbl']
                }
    test_data_frame = pd.DataFrame.from_dict(test_car)

    return test_data_frame
