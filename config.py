import os
from dotenv import load_dotenv

load_dotenv()

MYSQL_CONFIG = {
    'host': os.getenv('MYSQL_HOST', 'localhost'),
    'user': os.getenv('MYSQL_USER', 'root'),
    'password': os.getenv('MYSQL_PASSWORD', '02416.HK'),
    'database': os.getenv('MYSQL_DATABASE', 'order_service'),
    'charset': 'utf8mb4'
}

OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://192.168.8.50:11434/v1')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen2.5:7b')

CHROMA_PATH = os.getenv('CHROMA_PATH', './chroma_db')
CHROMA_COLLECTION = os.getenv('CHROMA_COLLECTION', 'orders')

EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'nomic-embed-text')
TOP_K = int(os.getenv('TOP_K', '5'))
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.5'))