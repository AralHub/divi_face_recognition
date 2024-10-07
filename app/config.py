import os

from pydantic_settings import BaseSettings
from dotenv import load_dotenv
load_dotenv()

class Settings(BaseSettings):
    mongodb_url: str = os.getenv('MONGO_URL')
    #mongodb_url: str = "mongodb://mongo:27017/face_recognition"
    database_name: str = "face_recognition"
    collections: str = ("collections")


    image: str = "pavel.png"
    UPLOAD_DIR: str = 'images'

settings = Settings()

