# MLflow Project

Este projeto utiliza **MLflow** para gerenciar o ciclo de vida de modelos de aprendizado de máquina, incluindo o rastreamento de experimentos, a gestão de modelos e a implantação de modelos em produção. O MLflow facilita a experimentação com diferentes algoritmos e parâmetros, e permite acompanhar e reproduzir os resultados com facilidade.

## Sumário

- [Pré-requisitos](#pré-requisitos)
- [Configuração do Ambiente](#configuração-do-ambiente)
- [Estrutura de Diretórios](#estrutura-de-diretórios)

## Pré-requisitos

Certifique-se de ter as seguintes dependências instaladas:

- [Python](https://www.python.org/) 3.8 ou superior
- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [Git](https://git-scm.com/)

## Configuração do Ambiente

### 1. Clone este repositório

Clone o repositório para o seu ambiente local:

```bash
git clone {URL_DO_REPOSITORIO}
cd docker_mlflow
```

### 2. Configuração do Docker

O MLflow usa o Docker para rodar o banco de dados de experimentos e o servidor web. O docker-compose.yml define os serviços necessários para o MLflow.

- Para iniciar o ambiente Docker e criar as imagens necessárias, execute o comando:

```bash
python create_and_start.py
```

## Estrutura de Diretórios

O projeto tem a seguinte estrutura de diretórios:

/mlflow
    Dockerfile
    docker-compose.yml
    create_and_start.py
    requirements.txt
    README.md

- **create_and_start.py**: Script Python para inicializar o banco de dados MLflow e - rodar o Docker.
- **requirements.txt**: Arquivo de dependências Python para o ambiente de - desenvolvimento.