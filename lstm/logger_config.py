import logging

# Configuração básica para o arquivo de log
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='app.log',
    filemode='a'
)

# Criar um logger nomeado
logger = logging.getLogger(__name__)

# Configuração do console handler para exibir mensagens no terminal
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    '%(asctime)s - %(filename)s - %(levelname)s - %(message)s'
)
console_handler.setFormatter(console_formatter)

# Adiciona o console handler ao root logger
logging.getLogger().addHandler(console_handler)


# Função para obter o logger configurado
def get_logger(name=__name__):
    return logging.getLogger(name)
