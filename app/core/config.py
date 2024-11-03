from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MONGODB_URL: str
    DATABASE_NAME: str
    UPLOAD_DIR: str
    MEDIA_URL: str
    COLLECTIONS: str
    MODEL_NAME: str = "buffalo_l"
    WORKER_POOL_SIZE: int = 4

    class Config:
        env_file = ".env"


settings = Settings()
