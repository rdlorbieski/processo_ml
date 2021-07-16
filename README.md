# Aplicação

O presente repositório tem como objetivo divulgar a solução do problema proposto no teste seletivo de pesquisador 2 de Machine Learning pelo Instituto Senai de Inovação em Sistemas Embarcados.

A base toda do projeto foi construída em cima da linguagem Python. Entretanto pode rodar independente do ambiente via docker.

## Pipeline da Solução

imagem

## Organização do Código

```
aplicacao-flask
├── algorithms 
│   └── challenge_senai
│       ├── predictor_test.py
│       ├── predictor_train.py
│       └── main.py
│       
├── dataset
│   ├── created
│   │   ├── df_test_validation.parquet
│   │   ├── df_train.parquet
│   │   └── df_trainer_test.parquet
│   │   
│   └── original
│       ├── X_test.csv
│       ├── X_train.csv
│       └── y_test.csv
│       
├── docs
│   ├── build
│   │	├── ...
│   │	└── html 
│   ├── images
│   └── source
│
├── repository
│   └── repository_service.py  
│
├── test
│   ├── __init__.py
│   └── test_sample.py
│
├── trained
│   └── models
│       ├── models_to_evaluate
│       │   ├── level_0
│       │   ├── level_1
│      	│   └── level_2
│       │
│       └── models_to_train
│           
├── app.py
├── config.py
├── docker-compose.yml
├── Dockerfile
├── logger
├── README.md
└── requirements.txt
```