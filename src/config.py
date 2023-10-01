"""Introduce data path constants and model parameters."""
from pathlib import Path

from sklearn.linear_model import LinearRegression

MCA_COMP = 11
# MCA_VIS_COMP = 2
PCA_COMP = 15
# PCA_VIS_COMP = 2

PROJECT_PATHNAME = Path(__file__).resolve().parent.parent
FILE_PATHNAME = PROJECT_PATHNAME / 'data/Automobile_data.csv'

CAT_ATTR = ("make", "fuel-type", "aspiration", "num-of-doors", "body-style",
            "drive-wheels", "engine-location", "engine-type",
            "num-of-cylinders", "fuel-system")
CAT_ORD_ATTR = ('normalized-losses', 'bore', 'stroke', 'horsepower',
                'peak-rpm', 'price')
NUM_ATTR = ("wheel-base", "length", "width", "height", "curb-weight",
            "engine-size", "compression-ratio", "city-mpg", "highway-mpg")
SYMBOLING = ("symboling",)


MODELS = {
         'LinearRegression': LinearRegression()
         }
