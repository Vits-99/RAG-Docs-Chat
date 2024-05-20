from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    def __init__(self):
        self.MODEL = os.getenv("MODEL")
        self.FILE_PATH = os.getenv("FILE_PATH")
        self.EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL")
        self.API_KEY = os.getenv("API_KEY")
        self.BASE_URL = os.getenv("BASE_URL")

settings = Settings()