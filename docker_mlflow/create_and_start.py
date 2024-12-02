import subprocess
import os


# Função para inicializar o Docker
def start_docker_compose():
    """Inicia o Docker Compose."""
    print("Iniciando Docker Compose...")
    subprocess.run(["docker-compose", "up", "--build"], check=True)


if __name__ == "__main__":
    # Verifica se o arquivo SQLite já existe, caso contrário cria o banco
    db_path = './mlflow.db'
    if not os.path.exists(db_path):
        open(db_path, 'w').close()
        print("Banco de dados SQLite mlflow.db criado com sucesso!")

    # Inicia o Docker
    start_docker_compose()
