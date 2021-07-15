import pandas as pd
import os
import config
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import KMeans
import warnings
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
import numpy as np
from algorithms.challenge_senai.SklearnHelper import SklearnHelper
import repository.repository_service as rs

pd.set_option('display.float_format', lambda x: '%.2f' % x)
warnings.filterwarnings("ignore")


def concatXY(x, y, nome):
    """
    function that merge features with target

    :param df: dataframe used
    """
    seriey = pd.Series(y, name=nome)
    df_train = pd.concat([x, seriey], axis=1)
    return df_train

def splitFeaturesTarget(df):
    """
    function that split features of target

    :param df: dataframe used
    """
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

def remove_columns_linearly_dependent_train_test(df_tr, df_te):
    """
    function that removes linearly dependent variables found in the test dataframe

    :return df_test_to_train: dataframe to generate the y_train
    :return df_test_real: dataframe to test the solution
    """

    # c1 has same variance that c5, c15
    # c2 has same variance that c8
    # c4 has same variance that c9
    # c6 has same variance that c7
    # c14 has same variance that c17

    # Create correlation matrix
    corr_matrix = df_te.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

    # Drop features
    df_train = df_tr.drop(to_drop, axis=1)
    df_test = df_te.drop(to_drop, axis=1)

    return df_train, df_test

def create_test_dataframes_to_train_and_validate():
    """
    function that creates the dataframe to generate the y_train and to generate the solution evaluation dataframe

    :return df_test_to_train: dataframe to generate the y_train
    :return df_test_real: dataframe to test the solution
    """

    x_train, x_test, y_test = rs.get_dfs_from_csv()
    print(x_train.shape[0], x_test.shape[0], y_test.shape[0])
    # print(x_train.isnull().sum())

    # =============== 1a abordagem: KMeans:
    clustering = KMeans(n_clusters=2, random_state=4).fit(x_train)
    # print(clustering.labels_)

    # =============== 2a abordagem: treinar com um pouco de dados de test:

    # remove columns with same variance
    x_train, x_test = remove_columns_linearly_dependent_train_test(x_train, x_test)

    df_teste = pd.concat([x_test, y_test], axis=1)
    df_teste['id'] = df_teste.index

    #  remove outliers para treinar com dados certos
    clf = EllipticEnvelope(support_fraction=1., contamination=0.2).fit(df_teste)
    df_teste_sem_out = df_teste.drop(df_teste[clf.predict(df_teste) == -1].index.values.tolist())

    df_teste_treino = df_teste_sem_out.sample(frac=.25)

    list_ids_a_remover = []  # pega o dataset original e ve quais ids tem que remover dali
    for indexes, row in df_teste.iterrows():
        if row.id in list(df_teste_treino.id):
            list_ids_a_remover.append(int(row.id))

    df_teste_real = df_teste.copy()
    df_teste_real = df_teste_real[~df_teste_real.id.isin(list_ids_a_remover)]

    df_teste_treino = df_teste_treino.drop(columns=['id'])
    df_teste_real = df_teste_real.drop(columns=['id'])
    return x_train, df_teste_treino, df_teste_real


def create_model_to_predict_y_train(df_teste_treino):
    """
    function that creates a model to predict y_train

    :param df_teste_treino: dataframe used to create a model responsible for predicting y_train
    """

    ada_params = {
        'n_estimators': 500,
        'learning_rate': 0.75
    }
    SEED = 4
    ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
    x, y = splitFeaturesTarget(df_teste_treino)
    print("O numero de linhas e colunas = ",x.shape)
    model = ada.fit(x, y)
    rs.save_model(model, "models_to_train", "adaboost")


def predict_y_train(x_train):
    y_train = []
    ada = rs.load_model("models_to_train","adaboost")
 #   print(ada.predict(array))
    for indexes, row in x_train.iterrows():
        y_train.append(ada.predict([row]))
    return y_train