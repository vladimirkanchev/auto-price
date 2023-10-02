"""Train model for prediction."""
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


def train_model(x_train: pd.DataFrame, y_train: pd.DataFrame, models: Dict):
    """Train the regression model on auto data."""
    trained_models = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        trained_models[name] = model

    return trained_models


def evaluate_model(x_val: pd.DataFrame, y_val: pd.DataFrame,
                   trained_models: Dict):
    """Evaluate the trained regression model with set-aside evaluation data."""
    model_name, mae, mse, rmse, r_2 = [], [], [], [], []

    for name, trained_model in trained_models.items():
        predictions = trained_model.predict(x_val)
        model_name.append(name)
        mae.append(mean_absolute_error(y_val, predictions))
        mse.append(mean_squared_error(y_val, predictions))
        rmse.append(np.sqrt(mean_squared_error(y_val, predictions)))
        r_2.append(r2_score(y_val, predictions))

    result = {'Model': model_name,
              'MAE': mae,
              'MSE': mse,
              'RMSE': rmse,
              'R2': r_2}

    return result


def inference_model(x_val: pd.DataFrame, trained_models: Dict):
    """Evaluate the trained regression model with set-aside evaluation data."""
    model_name, predicts = [], []

    for name, trained_model in trained_models.items():
        predict = trained_model.predict(x_val.values.reshape(-1, 11))
        model_name.append(name)
        predicts.append(predict)

    result = {'Model': model_name,
              'Predicts': predicts}

    return result
