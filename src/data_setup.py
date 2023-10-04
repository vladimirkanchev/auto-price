"""Load and save data operations."""
import pandas as pd
import config


def load_data() \
        -> pd.DataFrame:
    """Load raw auto data."""
    data_frame = pd.read_csv(config.DATA_PATHNAME)

    return data_frame
