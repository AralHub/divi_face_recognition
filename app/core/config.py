from pydantic_settings import BaseSettings


class S3Config(BaseSettings):
    aws_access_key_id: str
    aws_secret_access_key: str
    region_name: str
    endpoint_url: str  # Например, для Yandex Object Storage
    bucket_name: str

    class Config:
        env_file = ".env"


class Settings(BaseSettings):
    MONGODB_URL: str
    DATABASE_NAME: str
    UPLOAD_DIR: str
    MEDIA_URL: str
    COLLECTIONS: str
    MODEL_NAME: str = "buffalo_l"
    MODEL_PATH: str = "./models"
    WORKER_POOL_SIZE: int = 1
    s3_config = S3Config

    class Config:
        env_file = ".env"


settings = Settings()
