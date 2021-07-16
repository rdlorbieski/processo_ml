API - EXEMPLOS DE UTILIZAÇÃO
=============================

Este documento tem por objetivo documentar o código da API implementada
e como deve ser realizada a sua utilização.

Predict row
--------------------------------

Nesse módulo existe uma única chamada:
`localhost:5000/predict_row` .


*1 - Método POST:*

A requisição abaixo efetua a predição.

**Requisição:** localhost:5000/predict_row

**Exemplo de Entrada: (JSON)**

A entrada é um json com as 20 colunas:

.. code-block:: JSON

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

**Exemplo de Saída: (JSON)**

O retorno é um json com o resultado da variável alvo

.. code-block:: JSON

    {
        "target": 1.0
    }



Check Accuracy
--------------------------------

Nesse módulo existe uma única chamada:
`localhost:5000/check_accuracy` .


*1 - Método GET:*

A requisição abaixo efetua a verificação da acurácia do stacking.

**Requisição:** localhost:5000/check_accuracy

**Exemplo de Entrada: (JSON)**

Sem entrada

**Exemplo de Saída: (JSON)**

O retorno é um json com o resultado da variável alvo

.. code-block:: JSON

    {
            "accuracy": 0.7,
            "auc": 0.71
    }


Gerador de datasets e modelo para prever y_train
-------------------------------------------------

Nesse módulo existe uma única chamada:
`localhost:5000/generate_datasets_parquet_and_model_to_predict_ytrain` .


*1 - Método GET:*

A requisição abaixo gera os datasets necessários para a solução.
Além do modelo para gerar o y_train.

**Requisição:** localhost:5000/generate_datasets_parquet_and_model_to_predict_ytrain

**Exemplo de Entrada: (JSON)**

Sem entrada

**Exemplo de Saída: (JSON)**

O retorno é um json com o resultado da variável alvo

.. code-block:: JSON

    {
        "Response": "Datasets and Model generated with sucess!!"
    }



Gerador de modelos para avaliar a solução
-------------------------------------------------

Nesse módulo existe uma única chamada:
`localhost:5000/generate_models_to_validate_solution` .


*1 - Método GET:*

A requisição abaixo gera os modelos de machine learning necessários para a solução.

**Requisição:** localhost:5000/generate_models_to_validate_solution

**Exemplo de Entrada: (JSON)**

Sem entrada

**Exemplo de Saída: (JSON)**

O retorno é um json com o resultado da variável alvo

.. code-block:: JSON

    {
        "Response": "Models generated with success!!"
    }