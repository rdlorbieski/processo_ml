from unittest import TestCase
import algorithms.equation.equation_predictor as ep
from algorithms.equation.equation import Equation

class TestEquation_predictor(TestCase):

    def test_round_up(self):
        self.assertEqual(ep.round_up(2.1234, 2), 2.13)

    def test_calc_equation(self):
        equacao = Equation(a=1, b=-3, c=2)
        self.assertEqual(ep.calc_equation(equacao), (2.00, 1.00))
