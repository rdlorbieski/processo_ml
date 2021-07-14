import json
from algorithms.equation.equation_predictor import predictor_equation

json_string = """
    {
        "a": 1,
        "b": -3,
        "c": 2
    }
"""

data = json.loads(json_string)
equacao, x1, x2 = predictor_equation(data)

output = {
    "x1": x1,
    "x2": x2
}

print(equacao)
print(output)