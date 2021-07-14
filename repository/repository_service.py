import pickle
import os

import pandas as pd
import config
import logger as logger

# path where is trained models and loaders
path_dataset = config.path + os.path.sep + "datasets" + os.path.sep
path_model = config.path + os.path.sep + "trained" + os.path.sep + "models" + os.path.sep

# create path is not exist
if not os.path.exists(path_dataset):
    os.makedirs(path_dataset)
if not os.path.exists(path_model):
    os.makedirs(path_model)


def get_df_from_csv(name):
    """ returns filename as dataframe

    :param name: filename, example: name
    :return: dataframe in case of success or None in case of exception

    """

    try:
        filename = path_dataset + name + '.csv'
        df = pd.read_csv(filename, sep=';', engine='python')
        logger.log.info(filename + " loaded")
        return df
    except Exception as e:
        logger.log.info("RepositoryService get_df exception " + str(e))
        return None


def get_df_from_excel(name):
    """ returns filename as dataframe

    :param name: filename, example: name
    :return: dataframe in case of success or None in case of exception

    """

    try:
        filename = path_dataset + name + '.xlsx'
        df = pd.read_excel(filename, engine='openpyxl')
        logger.log.info(filename + " loaded")
        return df
    except Exception as e:
        logger.log.info("RepositoryService get_df exception " + str(e))
        return None


def save_model(model, name):
    """ save model to directory

    :param model: model to save
    :param name: name to save
    :return: True in case of success or False in case of Fail

    """

    try:
        filename = path_model + name + '.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
        logger.log.debug("saved model")
        return True
    except Exception as e:
        logger.log.debug("save_model exception:" + str(e))


def load_model(name):
    """ load model

    :param name: name to load
    :return: model in case of success or None in case of fail

    """

    try:
        filename = path_model + name + '.pkl'
        reg = pickle.load(open(filename, 'rb'))
        return reg
    except Exception as e:
        logger.log.debug("load_model exception:" + str(e))
        return None

def read_dataframe_parquet(name):
    """ read dataframe parquet

    :param name: name to load
    :return: model in case of success or None in case of fail

    """

    try:
        filename = path_model + name + '.parquet'
        return pd.read_parquet(filename)
    except Exception as e:
        logger.log.debug("read parquet exception:" + str(e))
        return None
