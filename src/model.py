"""Train model for prediction."""
# import logging
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

import config
from logger import logging
from preprocess import preprocess_transform


def train_and_predict_car_price(train_data_frame: pd.DataFrame,
                                test_data_frame: pd.DataFrame,
                                target: pd.Series) \
        -> List[float]:
    """Train the model on MCA transformed auto data with new car parameters."""
    x_train, x_test = preprocess_transform(train_data_frame,
                                           test_data_frame,
                                           transform=('mca', )
                                           )
    y_train = target
    trained_models = train_model(x_train, y_train,
                                 config.TYPEMODEL)
    result_info = inference_model(x_test, trained_models)

    return result_info


def train_model(x_train: pd.DataFrame,
                y_train: pd.DataFrame,
                models: Dict[str, Any]) \
        -> Dict[str, Any]:
    """Train the regression model on processed auto data."""
    trained_models = {}

    for name, model in models.items():
        model = model().fit(x_train, y_train)
        trained_models[name] = model

    logging.info("Train the price prediction model on auto dataset"
                 + " successfully.")

    return trained_models


def evaluate_model(x_val: pd.DataFrame,
                   y_val: pd.DataFrame,
                   trained_models: Dict[str, Any]) \
        -> Dict[str, List[str | List[float]]]:
    """Evaluate the trained regression model with set-aside evaluation data."""
    model_name, mae, mse, rmse, r_2 = [], [], [], [], []

    for name, trained_model in trained_models:
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

    logging.info("Evaluate the price prediction model on auto dataset"
                 + " successfully.")

    return result


def inference_model(test_val: pd.DataFrame,
                    trained_models: Dict[str, Any]) \
        -> Dict[str, List[str | List[float]]]:
    """Evaluate the trained regression model with set-aside evaluation data."""
    model_name, predicts = [], []

    for name, trained_model in trained_models.items():
        predict = trained_model.predict(test_val.values.reshape(-1, 11))
        model_name.append(name)
        predicts.append(predict)

    result = {'Model': model_name,
              'Predicts': predicts}
    print(result)
    logging.info("Compute the car price using the trained model"
                 + " successfully.")

    return result
