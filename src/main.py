"""Main function to start auto prediction algorithm."""
import data_setup as ds
import preprocess as pp
import model


def main():
    """Start auto prediction algorithm."""
    train_data_frame, target, cat_uniq_dict = pp.load_data_preprocess()

    # test_data_frame = ds.generate_test_car()

    test_data_frame = ds.random_generate_test_car(cat_uniq_dict)

    price = model.train_and_predict_car_price(train_data_frame,
                                              test_data_frame,
                                              target)

    print(f"Predicted car price: {price}")


if __name__ == "__main__":
    main()
