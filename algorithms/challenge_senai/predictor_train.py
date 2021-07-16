import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import KMeans
import warnings
import numpy as np
import repository.repository_service as rs
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split
import math

pd.set_option('display.float_format', lambda x: '%.2f' % x)
warnings.filterwarnings("ignore")


def round_up(n, decimals=0):
    """
    function to around

    :param n: number to around
    :param decimals: total of decimals
    :return result: rounded numeric value
    """
    if n == 0:
        return 0
    else:
        multiplier = 10 ** decimals
        return math.ceil(n * multiplier) / multiplier

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


def remove_outliers(cont, df):
    """
    function that remove outliers

    :return cont: contamination tax
    :return df: dataframe to apply the solution
    """
    if cont is None or cont == np.nan:
        cont = 0.2
    clf = EllipticEnvelope(support_fraction=1., contamination=cont).fit(df)
    df_test_sem_out = df.drop(df[clf.predict(df) == -1].index.values.tolist())
    return df_test_sem_out


def create_test_dataframes_to_train_and_validate():
    """
    function that separates the test set into a dataset to discover y_train and another to do the actual validation of
    the solution.

    :return df_trainer_test: dataframe to generate the y_train
    :return df_test_validation: dataframe to test the solution
    """

    x_train, x_test, y_test = rs.get_dfs_from_csv()
    print(x_train.shape[0], x_test.shape[0], y_test.shape[0])

    print(list(x_train.isnull().sum()))
    # remove any row with nan
    x_train = x_train.dropna(axis=0, how='any')

    df_test = pd.concat([x_test, y_test], axis=1)
    df_test['id'] = df_test.index

    #  remove outliers para treinar com dados certos
    df_test_sem_out = remove_outliers(0.2, df_test)
    df_trainer_test = df_test_sem_out.sample(frac=.40, random_state=1)

    list_ids_to_remove = []  # pega o dataset original e ve quais ids tem que remover dali
    for indexes, row in df_test.iterrows():
        if row.id in list(df_trainer_test.id):
            list_ids_to_remove.append(int(row.id))

    df_test_validation = df_test.copy()
    df_test_validation = df_test_validation[~df_test_validation.id.isin(list_ids_to_remove)]

    df_trainer_test = df_trainer_test.drop(columns=['id'])
    df_test_validation = df_test_validation.drop(columns=['id'])
    return x_train, df_trainer_test, df_test_validation


def train_model(classifier, df):
    """
    function that train a generic model with holdout with 20% to test

    :param classifier: classifier used to train the dataset
    :param df: dataframe used to create a model responsible for predicting y_train
    """
    x, y = splitFeaturesTarget(df)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    accuracy = round_up(accuracy_score(y_test, y_pred), 2)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    auc = round_up(metrics.auc(fpr, tpr), 2)

    return classifier, accuracy, auc

def show_performance(y_test, y_pred):
    """
    function that show performance comparing original target with predicted target

    :param y_test: original target
    :param y_pred: predicted target
    """
    accuracy = round_up(accuracy_score(y_test, y_pred), 2)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    auc = metrics.auc(fpr, tpr)
    cm = metrics.confusion_matrix(y_test, y_pred)
    return accuracy, auc, cm

def create_model_to_predict_y_train(df_test_train):
    """
    function that creates a model to predict y_train

    :param df_test_train: dataframe used to create a model responsible for predicting y_train
    """

    df_test_train_resume = df_test_train.loc[:, ['c1', 'c6', 'c8', 'c11', 'c13', 'c14','c16', 'c18', 'c19', 'target']]
    model = tree.DecisionTreeClassifier(max_depth=4)
    trained_model, accuracy, auc = train_model(model, df_test_train_resume)

    print("Acuracia da arvore de decisão:" + str(accuracy)+" auc = "+str(auc))
    return trained_model


def predict_y_train(x_train):
    """
    function that predict y_train

    :param x_train: features of train
    """

    x_train_resumed = x_train.loc[:, ['c1', 'c6', 'c8', 'c11', 'c13', 'c14','c16', 'c18', 'c19']]
    y_train = []
    tr = rs.load_model_to_train("tree")
    for indexes, row in x_train_resumed.iterrows():
        y_train.append(tr.predict([row])[0])
    return y_train
