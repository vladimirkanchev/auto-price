"""Main function to start auto prediction algorithm."""
import data_setup as ds
import preprocess as pp
import model
import utils


def main():
    """Start auto prediction algorithm."""
    train_data_frame, target, cat_uniq_dict = pp.load_data_preprocess()

    # test_data_frame = ds.generate_test_car()

    test_data_frame = ds.random_generate_test_car(cat_uniq_dict)

    price_info = model.train_and_predict_car_price(train_data_frame,
                                                   test_data_frame,
                                                   target)
    utils.log_price_info(test_data_frame, price_info)


if __name__ == "__main__":
    main()
