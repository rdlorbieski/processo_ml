# Aplicação

O presente repositório tem como objetivo divulgar a solução do problema proposto no teste seletivo de pesquisador 2 de Machine Learning pelo Instituto Senai de Inovação em Sistemas Embarcados.

A base toda do projeto foi construída em cima da linguagem Python. Entretanto pode rodar independente do ambiente via docker.

## Pipeline da Solução

imagem

## Organização do Código

```
aplicacao-flask
├── algorithms 
│   ├── alg1
│   │   ├── trainer.py
│   │   └── predictor.py 
│   └── ...
├── dataset
│   ├── database.csv
│   ├── instances
│   │   ├── instance-1.csv
│   │   ├── instance-2.csv
│   │   ├── instance-3.csv
│   │   └── ...
│   └── ...
├── docs
│   ├── build
│   │   ├── html
│	│	│	├── index.html
│	│	│	└── ...
│   │   └── ...
│   │
│   ├── images
│   │  
│   └── source
│
├── repository
│   └── repository_service.py  
│
├── test
│   ├── __init__.py
│   └── test_basic.py
│
├── trained
│   └── models
│   	├── model1.pkl
│   	├── model2.pkl
│   	└── ...
│
├── models
│   ├── model-features.csv
│   ├── model-features-simples.csv
│   └── ...
│
├── app.py
├── config.py
├── docker-compose.yml
├── Dockerfile
├── logger
├── README.md
└── requirements.txt
```