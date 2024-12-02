# Use uma imagem base do Python
FROM python:3.9-slim

# Definir o diretório de trabalho no container
WORKDIR /app

# Copiar o arquivo de requisitos, se houver
COPY requirements.txt /app/requirements.txt

# Instalar MLflow e quaisquer outras dependências necessárias
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expor a porta padrão do MLflow
EXPOSE 5000

# Adicionar um entrypoint para iniciar o servidor MLflow
ENTRYPOINT ["mlflow", "server"]

# Configurar as variáveis de ambiente do MLflow
# - `BACKEND_STORE_URI`: define o backend (ex.: sqlite, postgresql)
# - `ARTIFACT_ROOT`: define onde os artefatos serão armazenados
ENV BACKEND_STORE_URI sqlite:///mlflow.db
ENV ARTIFACT_ROOT /app/mlruns

# Criar um volume para armazenar os artefatos
VOLUME /app/mlruns

# Argumentos padrão para iniciar o servidor
CMD ["--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "/app/mlruns", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
