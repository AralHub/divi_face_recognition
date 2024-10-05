from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # mongodb_url: str = "mongodb://localhost:27017"
    mongodb_url: str = "mongodb://mongo:27017/face_recognition"
    database_name: str = "face_recognition"
    collections: str = ("collections")

    image: str = "pavel.png"
    UPLOAD_DIR: str = 'images'

settings = Settings()