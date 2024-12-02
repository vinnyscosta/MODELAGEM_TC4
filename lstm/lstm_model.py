import os
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import mlflow
import mlflow.keras
from .logger_config import get_logger
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.metrics import MeanSquaredError  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    LSTM,
    Dense,
    Input
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)

# Configuração do logger
logger = get_logger(__name__)


@tf.keras.utils.register_keras_serializable()
def mse(y_true, y_pred):
    return MeanSquaredError()(y_true, y_pred)


class LSTMModel:
    """Realiza o treinamento de um novo modelo LSTM para ações e
        permite predições a partir deles.
    """

    mlflow.set_tracking_uri("http://localhost:5000")

    def __init__(self, ticker: str, start_date: str, end_date: str) -> None:
        """Instancia o Objeto

        Args:
            ticker (str): Ação escolhida
            start_date (str): Data de inicio para a busca de dados
            end_date (str): Data fim para a busca de dados
        """

        mlflow.set_experiment(f"LSTM_{ticker}")

        self.ticker = ticker
        self.start_date_str = start_date
        self.end_date_str = end_date

    def load_data(self) -> None:
        """Carrega os dados da ação no yahoo finance utilizando
            as datas escolhidas.
        """
        self.df = yf.download(
            self.ticker,
            start=self.start_date_str,
            end=self.end_date_str
        )

    @staticmethod
    def create_window(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
        """Cria um janela com dados anterios para as datas do dataframe.

        Args:
            df (pd.DataFrame): Dataframe original com os dados
            n (int, optional): Tamanho da janela de dados anteriores.
                Defaults to 3.

        Returns:
            pd.DataFrame: Novo Dataframe com os dados já com
                suas respectivas janelas.
        """
        target_date = df.index[0] + timedelta(days=n)
        dates = []
        X, Y = [], []

        while True:
            df_subset = df.loc[:target_date].tail(n + 1)
            if len(df_subset) != n + 1:
                break

            values = df_subset['Close'].to_numpy()
            x, y = values[:-1], values[-1]
            dates.append(target_date)
            X.append(x)
            Y.append(y)

            next_week = df.loc[target_date:target_date + timedelta(days=7)]
            if len(next_week) < 2:
                break
            target_date = next_week.index[1]

        ret_df = pd.DataFrame({'Target Date': dates})
        X = np.array(X)
        for i in range(n):
            ret_df[f'Target-{n-i}'] = X[:, i]
        ret_df['Target'] = Y

        return ret_df

    @staticmethod
    def windowed_df_split(windowed_df: pd.DataFrame) -> tuple:
        """Divide os dados entre indice (dates), features (X) and target (Y)."

        Args:
            windowed_df (pd.DataFrame): Dataframe com janela de
                dados anteriores.

        Returns:
            tuple: _description_tuble contendo:
            - dates (np.ndarray): Indice dos dados
            - X (np.ndarray): Features dos dados
            - Y (np.ndarray): Target dos dados
        """
        df_np = windowed_df.to_numpy()
        dates = df_np[:, 0]
        X = df_np[:, 1:-1].reshape(-1, df_np.shape[1] - 2, 1).astype(np.float32)  # noqa
        Y = df_np[:, -1].flatten().astype(np.float32)
        return dates, X, Y

    def normalize_data(self) -> None:
        """Normaliza os dados com o MinMaxScaler.
        """
        self.df = self.df[['Close']]
        self.df.index = self.df.index.tz_localize(None)

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.df['Close'] = self.scaler.fit_transform(self.df[['Close']])

        self.windowed_df = self.create_window(self.df)

    def train_val_test_split(self) -> None:
        """Divide os dados entre treino, validação e teste.
        """
        self.dates, self.X, self.Y = self.windowed_df_split(self.windowed_df)
        q_80 = int(len(self.dates) * .8)
        q_90 = int(len(self.dates) * .9)

        # Dados de Treinamento
        self.dates_train = self.dates[:q_80]
        self.X_train = self.X[:q_80]
        self.y_train = self.Y[:q_80]

        # Dados de Validação
        self.dates_val = self.dates[q_80:q_90]
        self.X_val = self.X[q_80:q_90]
        self.y_val = self.Y[q_80:q_90]

        # Dados de Teste
        self.dates_test = self.dates[q_90:]
        self.X_test = self.X[q_90:]
        self.y_test = self.Y[q_90:]

    def train_model(self) -> None:
        """Treina o modelo LSTM.
        """
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        self.model = Sequential([
            Input((3, 1)),
            LSTM(64, return_sequences=True, input_shape=(
                self.X_train.shape[1], 1
            )),
            LSTM(64, return_sequences=False),
            Dense(32, activation='relu'),
            Dense(1)
        ])

        self.model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=0.001),
            metrics=['mean_absolute_error']
        )

        # Auto-logging do MlFlow
        mlflow.tensorflow.autolog()

        # Treinamento
        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=100,
            callbacks=[early_stopping],
            batch_size=32
        )

    def evaluate_model(self) -> None:
        """Executa o modelo com os dados de teste e calcula os erros.
        """
        predictions = self.model.predict(self.X_test)
        predictions = predictions.reshape(-1, 1)
        y_predictions = self.y_test.reshape(-1, 1)

        predictions = self.scaler.inverse_transform(predictions)
        y_predictions = self.scaler.inverse_transform(y_predictions)

        self.mae = mean_absolute_error(y_predictions, predictions)
        self.mse = mean_squared_error(y_predictions, predictions)
        self.rmse = np.sqrt(self.mse)
        self.mape = np.mean(
            np.abs((y_predictions - predictions) / y_predictions)
        ) * 100
        self.drift = np.abs(np.mean(y_predictions) - np.mean(predictions))
        self.error_variance = np.var(np.abs(y_predictions - predictions))

        # Log manual das métricas no MLflow
        mlflow.log_metric("MAE", self.mae)
        mlflow.log_metric("MSE", self.mse)
        mlflow.log_metric("RMSE", self.rmse)
        mlflow.log_metric("MAPE", self.mape)
        mlflow.log_metric("Drift", self.drift)
        mlflow.log_metric("Error Variance", self.error_variance)

        logger.info(f"Mean Absolute Error: {self.mae}")
        logger.info(f"Mean Squared Error: {self.mse}")
        logger.info(f"Root Mean Squared Error: {self.rmse}")
        logger.info(f"Mean Absolute Percentage Error: {self.mape}%")
        logger.info(f"Drift: {self.drift}")
        logger.info(f"Error Variance: {self.error_variance}")

    def save(self) -> None:
        """Salva o modelo treinado e o scaler utilizado.
        """
        self.folder = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        path = Path(f"./models/{self.ticker}/{self.folder}")
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o777)

        time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        model_filename = f"{path}/lstm_model_{time}.keras"
        scaler_filename = f"{path}/scaler_{time}.pkl"

        self.model.save(model_filename)
        joblib.dump(self.scaler, scaler_filename)

        mlflow.log_artifact(model_filename)
        mlflow.log_artifact(scaler_filename)

    def create_model(self) -> None:
        """Cria e treina o modelo LSTM.
        """
        time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        with mlflow.start_run(run_name=time):
            self.load_data()
            self.normalize_data()
            self.train_val_test_split()
            self.train_model()
            self.evaluate_model()
            self.save()

    @staticmethod
    def get_model_and_scaler(ticker: str = "AAPL"):
        """
        Retorna o modelo treinado e o scaler para um determinado ticker.

        Args:
            ticker (str, optional):
                Símbolo da ação para a qual o modelo e o scaler
                    devem ser carregados. O padrão é "AAPL".

        Raises:
            ValueError:
                Levantado se o experimento correspondente ao ticker não
                    for encontrado.
            FileNotFoundError:
                Levantado se o arquivo de scaler não estiver presente
                    nos artefatos.

        Returns:
            tuple:
                Um tuple contendo:
                    - `model`: O modelo treinado carregado do MLflow.
                    - `scaler`: O objeto scaler carregado do artefato.
        """
        # Busca pela run_id
        experiment_name = f"LSTM_{ticker}"
        experiment = mlflow.get_experiment_by_name(experiment_name)

        if experiment is None:
            raise ValueError(f"Experimento '{experiment_name}' não encontrado.")  # noqa

        experiment_id = experiment.experiment_id

        runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            order_by=["start_time desc"],
            max_results=1
        )
        run_id = runs['run_id'].values[0]

        # Carrega o modelo Keras do MLflow atraves da run
        model_uri = f'runs:/{run_id}/model'
        model = mlflow.tensorflow.load_model(model_uri)

        # Realiza o download dos artefatos para carregar o scaler
        artifact_path = mlflow.artifacts.download_artifacts(run_id=run_id)
        artifact_path = Path(artifact_path)

        # Busca pelo scaler
        scaler_file = None
        for file in os.listdir(artifact_path):
            if file.startswith("scaler") and file.endswith(".pkl"):
                scaler_file = artifact_path / file
                break

        # Erro caso não encontre o scaler
        if not scaler_file:
            raise FileNotFoundError("Arquivo de scaler não encontrado nos artefatos.")  # noqa

        scaler = joblib.load(scaler_file)

        return model, scaler

    @staticmethod
    def get_recent_data(ticker: str = "AAPL") -> list[float]:
        """_summary_

        Args:
            ticker (str, optional): Símbolo da ação para a qual o modelo e
                o scaler devem ser carregados. O padrão é "AAPL".

        Returns:
        list[float]:
            Lista contendo os preços de fechamento dos últimos 3 dias úteis.
        """
        data = yf.download(
            ticker,
            period="5d",
            interval="1d"
        )
        return data[-3:]["Close"].values.flatten()

    @classmethod
    def predict_latest_model_run(
        cls,
        days_ahead: int,
        ticker: str = "AAPL",
    ) -> dict:
        """_summary_

        Args:
            days_ahead (int): Numero de dias a serem preditos.
            ticker (str, optional): _description_. Defaults to "AAPL".

        Returns:
            dict: Dicionario com as predições.
        """

        model, scaler = cls.get_model_and_scaler(ticker)
        recent_data = cls.get_recent_data()
        predictions = {}
        print(recent_data)
        for i in range(days_ahead):

            # Transforma os dados de input
            to_predict = np.array(recent_data[-3:]).reshape(-1, 1)
            to_predict = scaler.fit_transform(to_predict)
            to_predict = to_predict.reshape(1, 3, 1)

            # Realiza a predição
            prediction = model.predict(to_predict)

            # Realiza a inversão do valor
            prediction = scaler.inverse_transform(prediction)
            predictions[f"Dia {i+1}"] = prediction[0, 0]
            recent_data = np.append(recent_data, prediction)

        return predictions


if __name__ == '__main__':

    # Uso do modelo
    modelo = LSTMModel("AAPL", '2023-01-01', '2024-11-13')
    modelo.create_model()

    # Predição com modelo mais recente
    predictions = LSTMModel.predict_latest_model_run(days_ahead=5)
    print(predictions)
