"""Main function to start auto prediction algorithm."""
from typing import Tuple

import pandas as pd
import streamlit as st

import config
from data_setup import load_data
from preprocess import preprocess_impute, preprocess_transform
from model import train_model, inference_model


def load_data_frame() \
        -> Tuple[pd.DataFrame, object]:
    """First toy example of auto prediction algorithm."""
    data_frame = load_data()
    data_frame = preprocess_impute(data_frame)

    data_frame.head(5)
    target = data_frame['price']
    data_frame = data_frame.drop('price', axis=1)

    return data_frame, target


def main():
    """Start auto prediction algorithm."""
    data_frame, target = load_data_frame()
    st.title("ML App for Car Prediction Price:")
    st.text("Enter the parameters of the car:")

    left_column, right_column = st.columns(2)
    with left_column:
        make = st.selectbox(
            'Car Brand:', ('audi', 'bmw', 'honda',
                           'chevrolet', 'dodge',
                           'isuzu', 'mazda', 'jaguar',
                           'mercedes-benz', 'mitsubishi'
                           'peugot', 'nissan', 'saab',
                           'toyota', 'volkswagen', 'volvo')
            )
        st.write('You selected:', make)

        fuel_type = st.selectbox(
            'Type of car fuel:', ('gas', 'diesel')
            )
        st.write('You selected:', fuel_type)

        aspiration = st.selectbox(
            'Type of car engine:', ('std', 'turbo')
            )
        st.write('You selected:', aspiration)

        num_of_doors = st.selectbox(
            'Number of car doors:', ('two', 'four')
            )
        st.write('You selected:', num_of_doors)

        body_style = st.selectbox(
            'Car body style:', ('convertible', 'hatchback',
                                'sedan', 'wagon', 'hardtop')
            )
        st.write('You selected:', body_style)

    with right_column:
        drive_wheels = st.selectbox(
            'Drive wheels:', ('rwd', '4wd', 'fwd')
            )
        st.write('You selected:', drive_wheels)

        engine_location = st.selectbox(
            'Car engine position:', ('front', 'rear')
            )
        st.write('You selected:', engine_location)

        engine_type = st.selectbox(
            'Type of engine:', ('dohc', 'ohcv', 'ohc',
                                'rotor')
            )
        st.write('You selected:', engine_type)

        num_of_cylinders = st.selectbox(
            'Number of engine cylinders:', ('two', 'three', 'four',
                                            'five', 'six', 'eight',
                                            'twelve')
            )
        st.write('You selected:', num_of_cylinders)

        fuel_system = st.selectbox(
            'Type of injection fuel:', ('1bbl', '2bbl', '4bbl',
                                        'mpfi', 'mfi', 'idi',
                                        'spdi')
            )
        st.write('You selected:', fuel_system)
    test_car = {'make': make, 'fuel-type': [fuel_type],
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

    x_train, x_test = preprocess_transform(data_frame, test_data_frame,
                                           transform=('mca', )
                                           )
    y_train = target
    trained_models = train_model(x_train, y_train, config.MODELS)

    result = inference_model(x_test, trained_models)
    price = round(result['Predicts'][0][0], 2)
    submit = st.button("Compute", type="primary")

    if submit:
        st.write('Market car price is: ' + str(price) + ' dollars')


if __name__ == "__main__":
    main()
