from flask import Flask, request, jsonify
import repository.repository_service as rs
import algorithms.challenge_senai.predictor_test as pte
import algorithms.challenge_senai.predictor_train as pt
from algorithms.challenge_senai.predictor_train import show_performance
from algorithms.challenge_senai.predictor_test import predictor_row
app = Flask(__name__)


@app.route('/')
def hello():
    return "welcome to the flask tutorials"

@app.route('/predict_row', methods=['POST'])
def predict_row():
    """
        returns json with value of target
        :param: json with parameters
        :example request:
       {
            "c1": -0.62,
            "c2": -1.63,
            "c3": -3.05,
            "c4": -1.37,
            "c5": -0.62,
            "c6": 0.82,
            "c7": 0.82,
            "c8": -1.63,
            "c9": -1.37,
            "c10": -2.52,
            "c11": -1.07,
            "c12": 0.00,
            "c13": -0.9,
            "c14": -1.01,
            "c15": -0.62,
            "c16": 0.02,
            "c17": -1.01,
            "c18": -0.74,
            "c19": 1.86,
            "c20": 5.28
        }
        :return: json with roots
        :example response:
        {
            "target": 1.0
        }

    """

    try:
        data = request.json
    except:
        return jsonify({"response": "Invalid Syntax"})

    if data is None:
        return jsonify({"response": "Expected data in request body"})

    target = predictor_row(data)

    return jsonify({
        "target": target,
        "response": "Ok"
    })


@app.route('/check_accuracy', methods=['GET'])
def check_performance_stacking():
    """
        returns json with value of accuracy and auc of model

        :return: json with accuracy and auc
        :example response:
        {
            "accuracy": 0.7,
            "auc": 0.71
        }
    """

    df_test_real = rs.get_dataset_test("df_test_validation")
    df_test_real = df_test_real.loc[:, ['c1', 'c6', 'c8', 'c11', 'c13', 'c14', 'c16', 'c18', 'c19', 'target']]
    df_test_real_normalized = pte.normalize_dataframe(df_test_real)
    x_true, y_true = pt.splitFeaturesTarget(df_test_real_normalized)
    stacking = rs.load_model("models_to_evaluate", "level_2", "stacking_ensemble_of_ensemble")
    y_pred = stacking.predict(x_true)
    acc, auc, mc = show_performance(y_true, y_pred)

    return jsonify({
        "acurácia": acc,
        "auc": auc,
        "response": "Ok"
    })

@app.route('/generate_datasets_parquet_and_model_to_predict_ytrain', methods=['GET'])
def generate_all_datasets_and_create_model_to_predict_ytrain():
    """
     Create all datasets (df_train full, df_trainer_test, df_test_validation) and model responsible to predict y_train

    """

    # ird_trainer
    x_train, df_teste_treino, df_teste_real = pt.create_test_dataframes_to_train_and_validate()
    tree_model = pt.create_model_to_predict_y_train(df_teste_treino)
    if not rs.file_exists("models_to_train", "tree"):
        rs.save_model_to_train(tree_model, "tree")

    if not rs.dataset_trainer_test_exists():
        rs.save_dataframe_test("df_trainer_test", df_teste_treino)

    if not rs.dataset_test_validation_exists():
        rs.save_dataframe_test("df_test_validation", df_teste_real)

    if not rs.dataset_train_exists():
        y_train = pt.predict_y_train(x_train)
        df_train = pt.concatXY(x_train, y_train, "target")
        df_train = df_train.dropna(axis=0, how='any')
        rs.save_dataframe_train(df_train)

    return jsonify({
        "Response": 'Datasets and Model generated with sucess!!'
    })


@app.route('/generate_models_to_validate_solution', methods=['GET'])
def generate_models_to_validate_solution():
    """
     Create all models (including stacking) to validate solution

    """
    df_train = rs.get_dataset_train()
    df_train = df_train.loc[:, ['c1', 'c6', 'c8', 'c11', 'c13', 'c14', 'c16', 'c18', 'c19', 'target']]
    df_train_clean = pt.remove_outliers(0.2, df_train)
    df_train_clean_normalized = pte.normalize_dataframe(df_train_clean)

    df_test_real = rs.get_dataset_test("df_test_validation")
    df_test_real = df_test_real.loc[:, ['c1', 'c6', 'c8', 'c11', 'c13', 'c14', 'c16', 'c18', 'c19', 'target']]
    df_test_real_normalized = pte.normalize_dataframe(df_test_real)

    pte.create_and_save_models(df_train_clean_normalized, df_test_real_normalized)

    return jsonify({
        "Response": 'Models generated with success!!'
    })



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
