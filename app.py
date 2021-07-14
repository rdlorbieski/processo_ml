from flask import Flask, request, jsonify

from algorithms.equation.equation_predictor import predictor_equation

app = Flask(__name__)


@app.route('/')
def hello():
    return "welcome to the flask tutorials"


@app.route('/predict_equation', methods=['POST'])
def predict_equation():
    """
        returns json with roots of quadratic equation

        :param: json with parameters
        :example request:
        {
            "a": 1,
            "b": -3,
            "c": 2
        }
        :return: json with roots
        :example response:
        {
            "x1": 2.0,
            "x2": 1.0
        }

    """

    try:
        data = request.json
    except:
        return jsonify({"response": "Invalid Syntax"})

    if data is None:
        return jsonify({"response": "Expected data in request body"})

    equacao, x1, x2 = predictor_equation(data)

    roots = {
        "x1": x1,
        "x2": x2
    }
    return jsonify({
        "response": "Ok",
        "roots": roots
    })


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
