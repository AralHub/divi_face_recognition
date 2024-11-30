import os

from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class S3Config(BaseSettings):
    aws_access_key_id: str = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = os.environ.get("AWS_SECRET_ACCESS_")
    endpoint_url: str = os.environ.get("ENDPOINT_URL")
    bucket_name: str = os.environ.get("BUCKET_NAME")
    server_name: str = os.environ.get("SERVER_NAME")


class Redis(BaseSettings):
    host: str = os.environ.get("REDIS_HOST")
    port: int = os.environ.get("REDIS_PORT")
    password: str = os.environ.get("REDIS_PASS")


class Settings(BaseSettings):
    MONGODB_URL: str = os.environ.get("MONGODB_URL")
    DATABASE_NAME: str = os.environ.get("DATABASE_NAME")
    COLLECTIONS: str = os.environ.get("COLLECTIONS")
    MODEL_NAME: str = "buffalo_l"
    MODEL_PATH: str = os.environ.get("MODEL_PATH")
    WORKER_POOL_SIZE: int = 1
    s3_config: S3Config = S3Config()  # Указание типа атрибута.
    redis_config: Redis = Redis()  # Указание типа атрибута.


settings = Settings()
