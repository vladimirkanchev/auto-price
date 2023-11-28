"""Introduce configu file with data path constants and model parameters."""
from datetime import datetime
import os
from pathlib import Path
from typing import Union

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

import config

TYPE = {
    "modelregressor": Union[LinearRegression, SVR]
}

GUI = {
    "nrows": 5,
    "ncols": 2
}

MESSAGE = {
    "0": "Car Brand:",
    "1": "Type of car fuel:",
    "2": "Type of car engine:",
    "3": "Number of car doors:",
    "4": "Car body style:",
    "5": "Drive wheels:",
    "6": "Car engine position:",
    "7": "Type of engine:",
    "8": "Number of engine cylinders:",
    "9": "Fuel system"
}

COMPONENT = {
    "mcacomponents": 11,
    "pcacomopnents": 15,
    "randomseed": 42
}

FILE = {
    "logfilename": f"{datetime.now().strftime('%m_d%d_%Y_%H_%M_%S')}.log",
    "modelfilename": "auto_price_predict.pkl",
    "datafilename": "Automobile_data.csv",
}
PATH = {
    "datapathname": Path(__file__).resolve().parent.parent / "data",
    "modelpathname": Path(__file__).resolve().parent.parent / "models",
    "logpathname": Path(__file__).resolve().parent.parent / ".logs"
}
os.makedirs(config.PATH['logpathname'], exist_ok=True)

ATTRIBUTE = {
    "catattr": ("make", "fuel-type", "aspiration", "num-of-doors",
                "body-style", "drive-wheels", "engine-location",
                "engine-type", "num-of-cylinders", "fuel-system"),
    "catordattr": ("normalized-losses", "bore", "stroke",
                   "horsepower", "peak-rpm", "price"),
    "numattr": ("wheel-base", "length", "width", "height",
                "curb-weight", "engine-size", "compression-ratio",
                "city-mpg", "highway-mpg"),
    "symb": ("symboling",),
    "target": ("price",)
}

MODEL = {
    'linearregression': LinearRegression()
}
