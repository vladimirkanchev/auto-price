"""An entrypoint file for price car prediction."""
from typing import Any, Dict, Tuple

import pandas as pd
import streamlit as st

#sys.path.append("src")
import config
from logger import logging
from model import train_and_predict_car_price
from preprocess import load_data_preprocess


def enter_car_parameters(cat_uniq_dict: Dict[str, Tuple[str]]) \
        -> pd.DataFrame:
    """Get test car paramters provide by user through App."""
    st.title("ML App for Car Prediction Price:")
    st.text("Enter the parameters of the car:")
    test_car = {}
    left_column, right_column = st.columns(config.GUI['ncols'])

    with left_column:
        for i in range(config.GUI['nrows']):
            attr = st.selectbox(
                config.MESSAGE[str(i)],
                cat_uniq_dict[config.ATTRIBUTE['catattr'][i]])
            st.write('You selected:', attr)
            test_car[config.ATTRIBUTE['catattr'][i]] \
                = [attr]

    with right_column:
        for i in range(config.GUI['nrows']):
            attr = st.selectbox(
                config.MESSAGE[str(i+config.GUI['nrows'])],
                cat_uniq_dict[config.ATTRIBUTE['catattr'][
                                               i+config.GUI['nrows']]])
            st.write('You selected:', attr)
            test_car[config.ATTRIBUTE['catattr'][i+config.GUI['nrows']]] \
                = [attr]

    test_data_frame = pd.DataFrame.from_dict(test_car)

    logging.info("User enter values for categorical attributes of a test car"
                 + " successfully.")

    return test_data_frame


def output_price(price_info: Dict[str, Any]) \
        -> None:
    """Output prince on App."""
    submit = st.button("Compute", type="primary")
    model_lst = price_info['Model']
    price_lst = price_info['Predicts']

    if submit:
        for model, price in zip(model_lst, price_lst):
            if hasattr(price[0], "__len__"):
                car_price = round(price[0][0], 2)
            else:
                car_price = round(price[0], 2)
                st.write('Market car price is: ' + str(car_price) +
                         ' dollars - ' + model)


def run() \
        -> None:
    """Run streamlit application with data from backend."""
    train_data_frame, target, cat_uniq_dict = load_data_preprocess()

    test_data_frame = enter_car_parameters(cat_uniq_dict)

    price_info = train_and_predict_car_price(train_data_frame,
                                             test_data_frame,
                                             target)
    output_price(price_info)


if __name__ == "__main__":
    run()
