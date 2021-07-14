API - EXEMPLOS DE UTILIZAÇÃO
=============================

Este documento tem por objetivo documentar o código da API implementada
e como deve ser realizada a sua utilização.

QE - Quadratic Equation
--------------------------------

No módulo do Equation existem uma única chamada:
`localhost:5000/predict_equation` .


*1 - Método POST:*

A requisição abaixo efetua a predição.

**Requisição:** localhost:5000/predict_equation

**Exemplo de Entrada: (JSON)**

A entrada é um json com os parâmetros a, b, c

.. code-block:: JSON

    {
        "a": 1,
        "b": -3,
        "c": 2
    }

**Exemplo de Saída: (JSON)**

O retorno é um json com as raízes da equação quadrática

.. code-block:: JSON

    {
        "x1": 2.0,
        "x2": 1.0
    }