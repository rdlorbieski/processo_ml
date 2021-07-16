import pickle
import os
import pyarrow

import pandas as pd
import config
import logger as logger

# path where is trained models and loaders
path_dataset = config.root_dir + os.path.sep + "datasets" + os.path.sep
path_model = config.root_dir + os.path.sep + "trained" + os.path.sep + "models" + os.path.sep

# create path is not exist
if not os.path.exists(path_dataset):
    os.makedirs(path_dataset)
if not os.path.exists(path_model):
    os.makedirs(path_model)


def get_dfs_from_csv():
    """
    return dataframes

    :return: x_train, x_test, y_test
    """

    try:
        column_names = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15',
                        'c16', 'c17', 'c18', 'c19', 'c20']
        x_train = pd.read_csv(path_dataset + "original" + os.path.sep + "X_train.csv", header=None, names=column_names)
        x_test = pd.read_csv(path_dataset + "original" + os.path.sep + "X_test.csv", header=None, names=column_names)
        y_test = pd.read_csv(path_dataset + "original" + os.path.sep + "y_test.csv", header=None, names=['target'])

        logger.log.info("file loaded")
        return x_train, x_test, y_test
    except Exception as e:
        logger.log.info("RepositoryService get_dfs exception " + str(e))
        return None

def save_model_to_train(model,  name):
    """
    save model to directory

    :param model: model to save
    :param folder: folder to save
    :param name: name to save
    :return: True in case of success or False in case of Fail

    """

    try:
        filename = path_model + "models_to_train" + os.path.sep + name + '.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
        logger.log.debug("saved model")
        return True
    except Exception as e:
        logger.log.debug("save_model exception:" + str(e))

def save_model(model, folder, level, name):
    """
    save model to directory

    :param model: model to save
    :param folder: folder to save
    :param name: name to save
    :return: True in case of success or False in case of Fail

    """

    try:
        filename = path_model + folder + os.path.sep + level + os.path.sep + name + '.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
        logger.log.debug("saved model")
        return True
    except Exception as e:
        logger.log.debug("save_model exception:" + str(e))

def load_model_to_train(name):
    """ load model

    :param name: name to load
    :return: model in case of success or None in case of fail

    """

    try:
        filename = path_model + "models_to_train" + os.path.sep + name + '.pkl'
        reg = pickle.load(open(filename, 'rb'))
        return reg
    except Exception as e:
        logger.log.debug("load_model exception:" + str(e))
        return None

def load_model(folder, level, name):
    """ load model

    :param name: name to load
    :return: model in case of success or None in case of fail

    """

    try:
        filename = path_model + folder + os.path.sep + level + os.path.sep + name + '.pkl'
        reg = pickle.load(open(filename, 'rb'))
        return reg
    except Exception as e:
        logger.log.debug("load_model exception:" + str(e))
        return None

def file_exists(folder, name):
    """ check if file with name of model exists

    :param model: model to save
    :param folder: folder to save
    :param name: name to save
    :return: true or false

    """

    try:
        filename = path_model + folder + os.path.sep + name + '.pkl'
        check_exists = os.path.isfile(filename)
        return check_exists
    except Exception as e:
        logger.log.debug("check error exception:" + str(e))
        return None


def save_dataframe_train(df_train):
    """
    save dataframe as parquet to directory

    :param df: dataframe to be save
    :return: True in case of success or False in case of Fail

    """

    try:

        filename = path_dataset + "created" + os.path.sep + "df_train.parquet"
        df_train.to_parquet(fname=filename)
        logger.log.debug("saved df train")
        return True
    except Exception as e:
        logger.log.debug("save_model exception:" + str(e))


def dataset_train_exists():
    """
    check if parquet file of dataframe train exists

    """

    try:
        filename = path_dataset + "created" + os.path.sep + "df_train.parquet"
        check_exists = os.path.isfile(filename)
        return check_exists
    except Exception as e:
        logger.log.debug("check error exception:" + str(e))
        return None


def get_dataset_train():
    """
       check if parquet file of dataframe train exists

    """
    filename = path_dataset + "created" + os.path.sep + "df_train.parquet"
    df = pd.read_parquet(filename)
    return df

def get_dataset_test(name):
    """
       check if parquet file of dataframe test exists

    """
    filename = path_dataset + "created" + os.path.sep + name + ".parquet"
    df = pd.read_parquet(filename)
    return df



def dataset_test_validation_exists():
    """
    check if parquet file of dataframe test exists

    """

    try:
        filename = path_dataset + "created" + os.path.sep + "df_test_validation.parquet"
        check_exists = os.path.isfile(filename)
        return check_exists
    except Exception as e:
        logger.log.debug("check error exception:" + str(e))
        return None

def dataset_trainer_test_exists():
    """
    check if parquet file of dataframe test exists

    """

    try:
        filename = path_dataset + "created" + os.path.sep + "df_trainer_test.parquet"
        check_exists = os.path.isfile(filename)
        return check_exists
    except Exception as e:
        logger.log.debug("check error exception:" + str(e))
        return None


def save_dataframe_test(name, df_teste_real):
    """
    save dataframe as parquet to directory

    :param df: dataframe to be save
    :return: True in case of success or False in case of Fail

    """

    try:

        filename = path_dataset + "created" + os.path.sep + name + ".parquet"
        df_teste_real.to_parquet(fname=filename)
        logger.log.debug("saved df test")
        return True
    except Exception as e:
        logger.log.debug("save_model exception:" + str(e))
