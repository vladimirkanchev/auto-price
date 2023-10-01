"""Load and save data operations."""
import pandas as pd
from config import FILE_PATHNAME


def load_data():
    """Load auto data."""
    data_frame = pd.read_csv(FILE_PATHNAME)
    return data_frame
