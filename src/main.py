"""Main function to start auto prediction algorithm."""
from sklearn.model_selection import train_test_split
import streamlit as st

import config
from data_setup import load_data
from preprocess import preprocess_impute, preprocess_transform
from utils import load_model, save_model
from model import train_model, evaluate_model


def toy_example():
    """First toy example of auto prediction algorithm."""
    data_frame = load_data()
    data_frame = preprocess_impute(data_frame)

    data_frame.head(5)
    target = data_frame['price']
    data_frame = data_frame.drop('price', axis=1)
    data_frame_tr = preprocess_transform(data_frame, transform=('pca', 'mca'))
    x_train, x_val, y_train, y_val = train_test_split(data_frame_tr,
                                                      target,
                                                      test_size=0.2,
                                                      random_state=42)

    trained_models = train_model(x_train, y_train, config.MODELS)
    result = evaluate_model(x_val, y_val, trained_models)
    print(result)
    save_model(trained_models['LinearRegression'])


def main():
    """Start auto prediction algorithm."""
    toy_example()
    st.title("Car Prediction Price App through ML:")
    st.text("Enter parameters of the car:")

    left_column, right_column = st.columns(2)
    with left_column:
        make = st.selectbox(
            'Car Brand:', ('audi', 'bmw', 'honda')
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

    submit = st.button("Compute", type="primary")
    if submit:
        st.write('Market car price is: ' + '1000' + ' dollars')


if __name__ == "__main__":
    main()
