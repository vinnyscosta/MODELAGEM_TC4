import mlflow
import mlflow.tensorflow
import yfinance as yf
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


mlflow.set_tracking_uri("http://localhost:5000")


class LSTMModel:
    def __init__(self, ticker: str, start_date: str, end_date: str):

        self.ticker = ticker
        self.start_date_str = start_date
        self.end_date_str = end_date

    def load_data(self):
        self.df = yf.download(
            self.ticker,
            start=self.start_date_str,
            end=self.end_date_str
        )

    @staticmethod
    def create_window(df: pd.DataFrame, n: int = 3):
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
    def windowed_df_split(windowed_df: pd.DataFrame):
        df_np = windowed_df.to_numpy()
        dates = df_np[:, 0]
        X = df_np[:, 1:-1].reshape(-1, df_np.shape[1] - 2, 1).astype(np.float32)
        Y = df_np[:, -1].astype(np.float32)
        return dates, X, Y

    def normalize_data(self):
        self.df = self.df[['Close']]
        self.df.index = self.df.index.tz_localize(None)

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.df['Close'] = self.scaler.fit_transform(self.df[['Close']])

        self.windowed_df = self.create_window(self.df)

    def train_val_test_split(self):
        self.dates, self.X, self.Y = self.windowed_df_split(self.windowed_df)
        q_80 = int(len(self.dates) * .8)
        q_90 = int(len(self.dates) * .9)

        self.dates_train, self.X_train, self.y_train = self.dates[:q_80], self.X[:q_80], self.Y[:q_80]
        self.dates_val, self.X_val, self.y_val = self.dates[q_80:q_90], self.X[q_80:q_90], self.Y[q_80:q_90]
        self.dates_test, self.X_test, self.y_test = self.dates[q_90:], self.X[q_90:], self.Y[q_90:]

    def train_model(self):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        self.model = Sequential([
            Input((3, 1)),
            LSTM(64, return_sequences=True, input_shape=(self.X_train.shape[1], 1)),
            LSTM(64, return_sequences=False),
            Dense(32, activation='relu'),
            Dense(1)
        ])

        self.model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=0.001),
            metrics=['mean_absolute_error']
        )

        # Inicia o auto-logging do MLflow
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

    def evaluate_model(self):
        predictions = self.model.predict(self.X_test)
        predictions = predictions.reshape(-1, 1)
        y_predictions = self.y_test.reshape(-1, 1)

        predictions = self.scaler.inverse_transform(predictions)
        y_predictions = self.scaler.inverse_transform(y_predictions)

        self.mae = mean_absolute_error(y_predictions, predictions)
        self.mse = mean_squared_error(y_predictions, predictions)
        self.rmse = np.sqrt(self.mse)
        self.mape = np.mean(np.abs((y_predictions - predictions) / y_predictions)) * 100

        print(f"Mean Absolute Error: {self.mae}")
        print(f"Mean Squared Error: {self.mse}")
        print(f"Root Mean Squared Error: {self.rmse}")
        print(f"Mean Absolute Percentage Error: {self.mape}%")

        # Loga manualmente as métricas no MLflow
        mlflow.log_metric("MAE", self.mae)
        mlflow.log_metric("MSE", self.mse)
        mlflow.log_metric("RMSE", self.rmse)
        mlflow.log_metric("MAPE", self.mape)

    def save(self):
        path = self.ticker
        path = Path(path)
        path.mkdir(exist_ok=True)
        path.chmod(0o777)

        time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        model_filename = f"{path}/lstm_model_{time}.h5"
        scaler_filename = f"{path}/scaler_{time}.pkl"

        self.model.save(model_filename)
        joblib.dump(self.scaler, scaler_filename)

        # Loga os arquivos no MLflow
        mlflow.log_artifact(model_filename)
        mlflow.log_artifact(scaler_filename)

    def create_model(self):
        time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        with mlflow.start_run(run_name=time):
            self.load_data()
            self.normalize_data()
            self.train_val_test_split()
            self.train_model()
            self.evaluate_model()
            self.save()


# Uso do modelo
modelo = LSTMModel("AAPL", '2023-01-01', '2024-11-13')
modelo.create_model()
