import math
from algorithms.equation.equation import Equation

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


def calc_equation(equacao):
    """
    calculate roots of quadratic equation

    :param equacao: object equation
    :return result: roots of quadratic equation
    """
    delta = (equacao.b*equacao.b) - (4 * equacao.a * equacao.c)
    x1 = (-equacao.b + math.sqrt(delta))/(2*equacao.a)
    x2 = (-equacao.b - math.sqrt(delta))/(2*equacao.a)

    return round_up(x1, 2), round_up(x2, 2)


def predictor_equation(json):
    """
    return prediction of json

    :param json: json with all items
    :return result: equation and roots of quadratic equation
    """
    equacao = Equation(
        a=json["a"],
        b=json["b"],
        c=json["c"]
    )

    x1, x2 = calc_equation(equacao)
    return equacao, x1, x2

