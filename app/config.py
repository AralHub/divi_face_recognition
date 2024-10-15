import os

from pathlib import Path
from pydantic.v1 import BaseModel
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent


class AuthJWT(BaseModel):
    private_key_path: Path = BASE_DIR / "keys" / "jwt-private.pem"
    public_key_path: Path = BASE_DIR / "keys" / "jwt-public.pem"
    algorithm: str = "RS256"


class Settings(BaseSettings):
    mongodb_url: str = os.getenv("MONGO_URL")
    # mongodb_url: str = "mongodb://mongo:27017/face_recognition"
    database_name: str = "face_recognition"
    collections: str = "collections"

    image: str = "pavel.png"
    UPLOAD_DIR: str = "images"

    auth_jwt: AuthJWT = AuthJWT()


settings = Settings()
