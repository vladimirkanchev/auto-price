"""Load, save and generate data operations."""
import pandas as pd
import config


def load_data() \
        -> pd.DataFrame:
    """Load raw auto data."""
    data_frame = pd.read_csv(config.DATA_PATHNAME)

    return data_frame


def random_generate_car() \
        -> pd.DataFrame:
    """Generate a test car with random parameter values met in auto dataset."""


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
