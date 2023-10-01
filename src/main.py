"""Main function to start auto prediction algorithm."""
import config
from data_setup import load_data
from preprocess import preprocess_impute, preprocess_transform
from train import train_model


def toy_example():
    """First toy example of auto prediction algorithm."""
    data_frame = load_data()

    data_frame = preprocess_impute(data_frame)
    data_frame_tr = preprocess_transform(data_frame, transform=('pca', 'mca'))
    print(data_frame_tr.shape)
    print(data_frame_tr.head(20))
    model = train_model(data_frame_tr, config.MODELS['LinearRegression'])


def main():
    """Start auto prediction algorithm."""
    toy_example()


if __name__ == "__main__":
    main()
