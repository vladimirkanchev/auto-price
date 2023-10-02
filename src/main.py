"""Main function to start auto prediction algorithm."""
from sklearn.model_selection import train_test_split

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


if __name__ == "__main__":
    main()
