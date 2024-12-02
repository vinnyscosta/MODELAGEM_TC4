# LSTM Model Para Predição de Preço de Ações

Este projeto implementa um modelo **LSTM (Long Short-Term Memory)** para previsão de preços de ações. Ele utiliza dados financeiros do Yahoo Finance e realiza o treinamento, avaliação e salvamento do modelo, com suporte ao rastreamento de experimentos através do **MLflow**.  

## Funcionalidades

- **Coleta de dados**:
  - Baixa dados de preços históricos de ações usando o [Yahoo Finance](https://finance.yahoo.com/).
- **Preparação de dados**:
  - Normalização com `MinMaxScaler`.
  - Criação de janelas temporais para os dados de entrada.
- **Treinamento do modelo**:
  - Treina um modelo LSTM com camadas totalmente conectadas.
  - Divisão automática dos dados em conjunto de treino, validação e teste.
  - Suporte a callbacks como `EarlyStopping`.
- **Avaliação do modelo**:
  - Métricas como **MAE**, **MSE**, **RMSE**, **MAPE** e **variância do erro**.
  - Registro automático de métricas no MLflow.
- **Salvamento de artefatos**:
  - O modelo treinado e o scaler são salvos localmente e no MLflow.
- **Previsão de preços futuros**:
  - Gera predições de dias futuros com base nos dados mais recentes.

---

## Configuração do Ambiente

### 1. Pré-requisitos

- Python 3.9 ou superior.
- Banco de dados MLflow configurado e acessível (por padrão, em `http://localhost:5000`).

### 2. Instalação

Clone o repositório e instale as dependências:

```bash
git clone {URL_DO_REPOSITORIO}
cd lstm
pip install -r requirements.txt
```

### 3. Dependências

- yfinance
- numpy
- pandas
- tensorflow
- joblib
- mlflow
- scikit-learn

## Estrutura do Projeto
```bash
lstm/
├── __init__.py   # Inicializações de classes e metodos.
├── lstm_model.py   # Classe principal para treinamento e previsão.
├── logger_config.py # Configuração do logger.
├── requirements.txt    # Dependências do projeto.
└── README.md           # Documentação do projeto.
```

## Uso

### 1. Inicialização do MLFlow

Siga o passo a passo do README.md do modulo 'docker_mlflow' em ./mlflow/README.md ou inicialize o modulo localmente executando:

```bash
pip install mlflow
mlflow ui
```

### 2. Treinamento de um Modelo
```python 
from lstm import LSTMModel

# Configuração
ticker = "AAPL"  # Exemplo: Apple Inc.
start_date = "2020-01-01"
end_date = "2023-12-31"

# Criação do modelo
model = LSTMModel(ticker, start_date, end_date)
model.create_model()
```

### 3. Previsão com o Modelo Mais Recente
```python 
from src.lstm_model import LSTMModel

# Previsão para os próximos 5 dias
predictions = LSTMModel.predict_latest_model_run(days_ahead=5, ticker="AAPL")
print(predictions)
```

## Métricas Registradas

Durante o treinamento, as seguintes métricas são registradas:

- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- Drift (deslocamento médio entre valores reais e preditos)
- Error Variance (variância dos erros)
