"""An entrypoint file for price car prediction."""
import sys
from typing import Dict, Tuple

import pandas as pd
import streamlit as st

sys.path.append("src")
from model import train_and_predict_car_price
from preprocess import load_data_preprocess


def enter_car_parameters(cat_uniq_dict: Dict[str, Tuple[str]]) \
        -> pd.DataFrame:
    """Get test car paramters provide by user through App."""
    st.title("ML App for Car Prediction Price:")
    st.text("Enter the parameters of the car:")

    left_column, right_column = st.columns(2)
    with left_column:
        make = st.selectbox(
            'Car Brand:', cat_uniq_dict['make'])
        st.write('You selected:', make)

        fuel_type = st.selectbox(
            'Type of car fuel:', cat_uniq_dict['fuel-type'])
        st.write('You selected:', fuel_type)

        aspiration = st.selectbox(
            'Type of car engine:', cat_uniq_dict['aspiration'])
        st.write('You selected:', aspiration)

        num_of_doors = st.selectbox(
            'Number of car doors:', cat_uniq_dict['num-of-doors'])
        st.write('You selected:', num_of_doors)

        body_style = st.selectbox(
            'Car body style:', cat_uniq_dict['body-style'])
        st.write('You selected:', body_style)

    with right_column:
        drive_wheels = st.selectbox(
            'Drive wheels:', cat_uniq_dict['drive-wheels'])
        st.write('You selected:', drive_wheels)

        engine_location = st.selectbox(
            'Car engine position:', cat_uniq_dict['engine-location'])
        st.write('You selected:', engine_location)

        engine_type = st.selectbox(
            'Type of engine:', cat_uniq_dict['engine-type'])
        st.write('You selected:', engine_type)

        num_of_cylinders = st.selectbox(
            'Number of engine cylinders:', cat_uniq_dict['num-of-cylinders'])
        st.write('You selected:', num_of_cylinders)

        fuel_system = st.selectbox(
            'Type of injection fuel:', cat_uniq_dict['fuel-system'])
        st.write('You selected:', fuel_system)

        test_car = {'make': make,
                    'fuel-type': [fuel_type],
                    'aspiration': [aspiration],
                    'num-of-doors': [num_of_doors],
                    'body-style': [body_style],
                    'drive-wheels': [drive_wheels],
                    'engine-location': [engine_location],
                    'engine-type': [engine_type],
                    'num-of-cylinders': [num_of_cylinders],
                    'fuel-system': [fuel_system]
                    }

        test_data_frame = pd.DataFrame.from_dict(test_car)

        return test_data_frame


def output_price(price: float) \
        -> None:
    """Output prince on App."""
    submit = st.button("Compute", type="primary")

    if submit:
        st.write('Market car price is: ' + str(price) + ' dollars')


def run() \
        -> None:
    """Run streamlit application with data from backend."""
    train_data_frame, target, cat_uniq_dict = load_data_preprocess()

    test_data_frame = enter_car_parameters(cat_uniq_dict)

    price = train_and_predict_car_price(train_data_frame,
                                        test_data_frame,
                                        target)
    output_price(price)


if __name__ == "__main__":
    run()
