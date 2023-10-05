"""Preprocess data for training."""
import itertools
from typing import Tuple

import pandas as pd
import prince

import config
from data_setup import load_data
import utils
# from utils import replace_missing, mean_imputation,
# mode_imputation, convert_cat_ord_to_num

def load_data_preprocess() \
        -> Tuple[pd.DataFrame, object]:
    """First toy example of auto prediction algorithm."""
    data_frame = load_data()
    data_frame = preprocess_impute(data_frame)

    target = data_frame['price']
    data_frame = data_frame.drop('price', axis=1)
    cat_unq_values = utils.get_unique_cat_values(data_frame)

    return data_frame, target, cat_unq_values


def preprocess_impute(data_frame: pd.DataFrame) \
        -> pd.DataFrame:
    """Find out and impute missing values in the loaded auto data."""
    data_frame = utils.replace_missing(data_frame)
    data_frame = utils.mean_imputation(data_frame, config.NUM_ATTR)
    data_frame = utils.mode_imputation(data_frame, config.CAT_ATTR)
    data_frame = utils.zero_imputation(data_frame, config.CAT_ORD_ATTR)

    data_frame = utils.convert_cat_ord_to_num(data_frame)

    data_frame["symboling"] = data_frame["symboling"].astype(str)
    data_frame = utils.mode_imputation(data_frame, config.SYMBOLING)

    return data_frame


def preprocess_transform(train_data_frame: pd.DataFrame,
                         test_data_frame: pd.DataFrame,
                         transform: Tuple = 'None') \
        -> Tuple[pd.DataFrame, pd.Series]:
    """Apply PCA and MCA on numerical and categorical data respectively."""
    if transform is not None \
            and 'mca' not in transform \
            and 'pca' not in transform:
        raise ValueError("Transform must of type mca, pca, or both")

    if transform is None:
        return train_data_frame, test_data_frame

    if 'mca' in transform:
        cat_attr_new = tuple(itertools.chain(config.CAT_ATTR,
                                             config.SYMBOLING))
        cat_data_tr = preprocess_cat_data_mca(train_data_frame,
                                              test_data_frame,
                                              cat_attr_new)
        if len(transform) == 1:
            train_data_tr = cat_data_tr.iloc[:-1]
            test_data_tr = cat_data_tr.iloc[-1]
            return train_data_tr, test_data_tr

    if 'pca' in transform:
        num_attr_list = list(itertools.chain(config.NUM_ATTR,
                                             config.CAT_ORD_ATTR))
        num_attr = tuple(num_attr_list.remove('price'))
        num_data_tr = preprocess_num_data_pca(train_data_frame,
                                              test_data_frame,
                                              num_attr)
        if len(transform) == 1:
            train_data_tr = num_data_tr.iloc[:-1]
            test_data_tr = num_data_tr.iloc[-1]
            return train_data_tr, test_data_tr

    data_tr = pd.concat([cat_data_tr, num_data_tr], axis=1)
    train_data_tr = data_tr.iloc[:-1]
    test_data_tr = data_tr.iloc[-1]
    # data_tr = pd.concat(data_tr_lst, ignore_index=True, axis=1)

    return train_data_tr, test_data_tr


def preprocess_cat_data_mca(train_data_frame: pd.DataFrame,
                            test_data_frame: pd.DataFrame,
                            cat_attrs: Tuple) \
        -> pd.DataFrame:
    """Apply MCA on categorical attributes to use for auto price prediction."""
    mca = prince.MCA(n_components=config.MCA_COMP,
                     random_state=config.RANDOM_SEED)
    # get principal components
    data_frame_mca = train_data_frame[list(cat_attrs)]
    data_frame_mca = pd.concat([data_frame_mca, test_data_frame], axis=0)
    mca_fit = mca.fit(data_frame_mca)

    # mca_fit.eigenvalues_summary
    data_cat_attr_mca = mca_fit.transform(data_frame_mca)

    return data_cat_attr_mca


def preprocess_num_data_pca(train_data_frame: pd.DataFrame,
                            test_data_frame: pd.DataFrame,
                            num_attrs: Tuple) \
        -> pd.DataFrame:
    """Apply PCA "on numerical attributes to use for auto price prediction."""
    pca = prince.PCA(n_components=config.PCA_COMP,
                     random_state=config.RANDOM_SEED)
    # get princical components
    data_frame_pca = train_data_frame[list(num_attrs)]
    data_frame_pca = pd.concat([data_frame_pca, test_data_frame], axis=0)

    pca_fit = pca.fit(data_frame_pca)
    # pca_fit.eigenvalues_summary
    data_num_attr_pca = pca_fit.transform(data_frame_pca)

    return data_num_attr_pca
