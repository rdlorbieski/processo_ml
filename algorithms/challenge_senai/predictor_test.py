from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier, StackingClassifier, VotingClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import repository.repository_service as rs
import algorithms.challenge_senai.predictor_train as pt
from sklearn.model_selection import KFold, cross_val_score
import pandas as pd
import numpy as np

def normalize_dataframe(df):
    df_bkp = df.copy()
    normalized_df = (df-df.min())/(df.max()-df.min())
    normalized_df['target'] = df_bkp['target']
    return normalized_df


def get_config_ensemble_nn():
    clf = BaggingClassifier(MLPClassifier(),max_samples=0.5, max_features=0.5)
    return clf

def get_config_ensemble_tree():
    clf = ExtraTreesClassifier(max_depth=5)
    return clf


def get_config_ensemble_lazy():
    knn1 = KNeighborsClassifier(n_neighbors=3, weights="uniform", p=1)  # p=1 manhattan_distance
    knn2 = KNeighborsClassifier(n_neighbors=3, weights="uniform", p=2)  # p=1 manhattan_distance
    knn3 = KNeighborsClassifier(n_neighbors=5, weights="distance", p=1)  # p=2 euclidean_distance
    knn4 = KNeighborsClassifier(n_neighbors=5, weights="distance", p=2)  # p=2 euclidean_distance

    estimator_list = [
        ('knn1', knn1),
        ('knn2', knn2),
        ('knn3', knn3),
        ('knn4', knn4)
    ]

    # Build stack model
    stack_model_knn = VotingClassifier(
        estimators=estimator_list, voting='soft'
    )
    return stack_model_knn


def get_config_ensemble_svm():
    svm1 = svm.SVC(kernel='linear', C=1.0, probability=True)
    clf = AdaBoostClassifier(n_estimators=20, base_estimator=svm1, algorithm='SAMME')
    return clf



def generate_prediction(x, y, x_test, model):
    model = model.fit(x, y)
    y_pred = model.predict(x_test)
    return y_pred

def avaliate_level_0(df_train_clean_normalized, df_test_real_normalized):
    performance_level_0 = {}
    names = ['knn1', 'knn2', 'knn3', 'knn4', 'rf1', 'dt1', 'etc', 'svm1', 'svm2', 'svm3']
    print("============================= LAZY =============================")
    x, y = pt.splitFeaturesTarget(df_train_clean_normalized)
    x_true, y_true = pt.splitFeaturesTarget(df_test_real_normalized)

    knn1 = KNeighborsClassifier(n_neighbors=3, weights="uniform", p=1)  # p=1 manhattan_distance
    knn2 = KNeighborsClassifier(n_neighbors=3, weights="uniform", p=2)  # p=1 manhattan_distance
    knn3 = KNeighborsClassifier(n_neighbors=5, weights="distance", p=1)  # p=2 euclidean_distance
    knn4 = KNeighborsClassifier(n_neighbors=5, weights="distance", p=2)  # p=2 euclidean_distance

    accuracy, auc, cm = pt.show_performance(y_true, generate_prediction(x, y, x_true, knn1))
    performance_level_0.update({'knn1': [accuracy, auc, cm]})
    accuracy, auc, cm = pt.show_performance(y_true, generate_prediction(x, y, x_true, knn2))
    performance_level_0.update({'knn2': [accuracy, auc, cm]})
    accuracy, auc, cm = pt.show_performance(y_true, generate_prediction(x, y, x_true, knn3))
    performance_level_0.update({'knn3': [accuracy, auc, cm]})
    accuracy, auc, cm = pt.show_performance(y_true, generate_prediction(x, y, x_true, knn4))
    performance_level_0.update({'knn4': [accuracy, auc, cm]})

    rf1 = RandomForestClassifier(max_depth=5, n_estimators=200, max_features="auto")
    dt1 = DecisionTreeClassifier(max_depth=5)
    etc = ExtraTreesClassifier(max_depth=5)

    accuracy, auc, cm = pt.show_performance(y_true, generate_prediction(x, y, x_true, rf1))
    performance_level_0.update({'rf1': [accuracy, auc, cm]})
    accuracy, auc, cm = pt.show_performance(y_true, generate_prediction(x, y, x_true, dt1))
    performance_level_0.update({'dt1': [accuracy, auc, cm]})
    accuracy, auc, cm = pt.show_performance(y_true, generate_prediction(x, y, x_true, etc))
    performance_level_0.update({'etc': [accuracy, auc, cm]})

    svm1 = svm.SVC(kernel='linear', C=1.0)
    svm2 = svm.SVC(kernel='rbf', C=1.0)
    svm3 = svm.SVC(kernel='poly', C=1.0)


    accuracy, auc, cm = pt.show_performance(y_true, generate_prediction(x, y, x_true, svm1))
    performance_level_0.update({'svm1': [accuracy, auc, cm]})
    accuracy, auc, cm = pt.show_performance(y_true, generate_prediction(x, y, x_true, svm2))
    performance_level_0.update({'svm2': [accuracy, auc, cm]})
    accuracy, auc, cm = pt.show_performance(y_true, generate_prediction(x, y, x_true, svm3))
    performance_level_0.update({'svm3': [accuracy, auc, cm]})

    lda = LinearDiscriminantAnalysis()
    accuracy, auc, cm = pt.show_performance(y_true, generate_prediction(x, y, x_true, lda))
    performance_level_0.update({'lda': [accuracy, auc, cm]})

    bayes = GaussianNB()
    accuracy, auc, cm = pt.show_performance(y_true, generate_prediction(x, y, x_true, bayes))
    performance_level_0.update({'bayes': [accuracy, auc, cm]})

    mlp = MLPClassifier()
    accuracy, auc, cm = pt.show_performance(y_true, generate_prediction(x, y, x_true, mlp))
    performance_level_0.update({'mlp': [accuracy, auc, cm]})

    classifiers_level_0 = [knn1, knn2, knn3, knn4, svm1, etc, mlp]
    return classifiers_level_0, performance_level_0


def avaliate_level_1(df_train_clean_normalized, df_test_real_normalized):
    performance_level_1 = {}
    x, y = pt.splitFeaturesTarget(df_train_clean_normalized)
    x_true, y_true = pt.splitFeaturesTarget(df_test_real_normalized)

    ensemble_lazy = get_config_ensemble_lazy()
    accuracy, auc, cm = pt.show_performance(y_true, generate_prediction(x, y, x_true, ensemble_lazy))
    performance_level_1.update({'ensemble_lazy': [accuracy, auc, cm]})

    ensemble_tree = get_config_ensemble_tree()
    accuracy, auc, cm = pt.show_performance(y_true, generate_prediction(x, y, x_true, ensemble_tree))
    performance_level_1.update({'ensemble_tree': [accuracy, auc, cm]})

    ensemble_nn = get_config_ensemble_nn()
    accuracy, auc, cm = pt.show_performance(y_true, generate_prediction(x, y, x_true, ensemble_nn))
    performance_level_1.update({'ensemble_nn': [accuracy, auc, cm]})

    ensemble_svm = get_config_ensemble_svm()
    accuracy, auc, cm = pt.show_performance(y_true, generate_prediction(x, y, x_true, ensemble_svm))
    performance_level_1.update({'ensemble_svm': [accuracy, auc, cm]})

    classifiers_level_1 = [ensemble_lazy, ensemble_tree, ensemble_nn, ensemble_svm]
    return classifiers_level_1, performance_level_1



def avaliate_level_2(df_train_clean_normalized, df_test_real_normalized):
    x, y = pt.splitFeaturesTarget(df_train_clean_normalized)
    x_true, y_true = pt.splitFeaturesTarget(df_test_real_normalized)

    ensemble_lazy = get_config_ensemble_lazy()
    ensemble_tree = get_config_ensemble_tree()
    ensemble_nn = get_config_ensemble_nn()
    ensemble_svm = get_config_ensemble_svm()
    estimator_list = [
        ('ensemble_lazy', ensemble_lazy),
        ('ensemble_tree', ensemble_tree),
        ('ensemble_nn', ensemble_nn),
        ('ensemble_svm', ensemble_svm)
    ]
    stack_model_svm = StackingClassifier(
        estimators=estimator_list, final_estimator=GaussianNB()
    )


    accuracy, auc, cm = pt.show_performance(y_true, generate_prediction(x, y, x_true, stack_model_svm))
    return stack_model_svm, accuracy, auc, cm

def predictor_row(json):


    c1 = json["c1"]
    c2 = json["c2"]
    c3 = json["c3"]
    c4 = json["c4"]
    c5 = json["c5"]
    c6 = json["c6"]
    c7 = json["c7"]
    c8 = json["c8"]
    c9 = json["c9"]
    c10 = json["c10"]
    c11 = json["c11"]
    c12 = json["c12"]
    c13 = json["c13"]
    c14 = json["c14"]
    c15 = json["c15"]
    c16 = json["c16"]
    c17 = json["c17"]
    c18 = json["c18"]
    c19 = json["c19"]
    c20 = json["c20"]

    column_names = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15',
                    'c16', 'c17', 'c18', 'c19', 'c20']
    row = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20]
    df2 = pd.DataFrame(np.array([row]), columns=column_names)
    df2 = df2.loc[:, ['c1', 'c6', 'c8', 'c11', 'c13', 'c14', 'c16', 'c18', 'c19']]
    row = [df2.iloc[0]]
    stacking = rs.load_model("models_to_evaluate", "level_2", "stacking_ensemble_of_ensemble")
    target = stacking.predict(row)[0]
    return target
