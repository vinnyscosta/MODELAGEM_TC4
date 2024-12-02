# Previsão de Preços de Ações com LSTM e MLflow

Este repositório implementa um sistema completo para previsão de preços de ações utilizando LSTM (Long Short-Term Memory) e MLflow para rastreamento de experimentos. Ele é dividido em dois módulos principais: mlflow (configuração e gerenciamento de experimentos) e lstm (treinamento e utilização do modelo de previsão). Cada módulo possui seu próprio README detalhado.

## Estrutura do Projeto

```bash
projeto-previsao-acoes/
├── lstm/               # Módulo principal para treinamento e previsão de preços de ações.
│   ├── __init__.py     # Inicialização do pacote LSTM.
│   ├── lstm_model.py   # Classe principal para treinamento e previsão.
│   ├── logger_config.py # Configuração do logger.
│   ├── requirements.txt # Dependências específicas do módulo lstm.
│   └── README.md       # Documentação do módulo lstm.
├── docker_mlflow/             # Módulo para configuração e gerenciamento de experimentos MLflow.
│   ├── docker-compose.yml # Configuração para execução do MLflow via Docker.
│   ├── create_and_start.py # Script para criar o banco e iniciar o MLflow.
│   ├── requirements.txt    # Dependências específicas do módulo mlflow.
│   └── README.md       # Documentação do módulo mlflow.
└── README.md           # Este arquivo. Documentação principal do projeto.
```

## Visão Geral dos Módulos

### 1. Módulo mlflow

Responsável por configurar e gerenciar o ambiente do MLflow, incluindo:

- Configuração do banco de dados para rastreamento de experimentos.
- Scripts para automação de setup e inicialização do servidor MLflow.
- Suporte a execução via Docker Compose.

Para mais detalhes sobre a configuração e uso do MLflow, veja o arquivo README.md do módulo mlflow.

### 2. Módulo lstm

Focado no desenvolvimento do modelo LSTM para previsão de preços de ações, com funcionalidades como:

- Coleta e preparação de dados históricos de ações.
- Treinamento, avaliação e registro de métricas no MLflow.
- Salvamento de modelos e scalers para uso futuro.
- Previsão de preços de ações com base em dados recentes.

Para mais detalhes sobre o modelo e uso do módulo lstm, veja o arquivo README.md do módulo lstm.

## Configuração do Ambiente

### 1. Pré-requisitos

- Python 3.9 ou superior.
- Docker e Docker Compose instalados no sistema.

### 2. Configuração dos Módulos
#### Configurar o módulo mlflow:

- Consulte o README.md do módulo mlflow para configurar o ambiente MLflow e iniciar o servidor.

#### Configurar o módulo lstm:

- Consulte o README.md do módulo lstm para configurar e treinar o modelo LSTM.

## Como Usar

1. Iniciar o MLflow: Certifique-se de que o servidor MLflow está rodando, conforme descrito no README do módulo mlflow.

2. Treinar o Modelo: Use o módulo lstm para coletar dados, treinar o modelo e registrar os experimentos no MLflow.

3. Consultar Resultados no MLflow: Acesse o servidor MLflow para visualizar os experimentos, métricas e artefatos gerados pelo treinamento.