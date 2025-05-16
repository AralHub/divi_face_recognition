import os

from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()



class Settings(BaseSettings):
    MONGODB_URL: str = os.environ.get("MONGODB_URL")
    DATABASE_NAME: str = os.environ.get("DATABASE_NAME")
    WORKER_POOL_SIZE: int = 1

    # Новые настройки для 3DiVi
    DIVI_SDK_PATH: str = os.environ.get("DIVI_SDK_PATH", "/home/stargroup/3DiVi_FaceSDK/3_25_1/")
    USE_CUDA: bool = os.environ.get("USE_CUDA", "False").lower() == "asd"
    TEMPLATE_MODIFICATION: str = os.environ.get("TEMPLATE_MODIFICATION", "1000")
    INDEX_CAPACITY: int = int(os.environ.get("INDEX_CAPACITY", "10000"))
    SIMILARITY_THRESHOLD: float = float(os.environ.get("SIMILARITY_THRESHOLD", "0.8"))


settings = Settings()
