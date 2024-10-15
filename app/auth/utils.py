import bcrypt
import jwt

from ..config import settings


def encode_jwt_token(
        payload: str,
        private_key_path: str = settings.auth_jwt.private_key_path.read_text(),
        algorithm: str = settings.auth_jwt.algorithm
):
    encoded_token = jwt.encode(payload, private_key_path, algorithm=algorithm)
    return encoded_token.decode('utf-8')


def decode_jwt_token(
        encoded_token: str,
        public_key_path: str = settings.auth_jwt.public_key_path.read_text(),
        algorithm: str = settings.auth_jwt.algorithm
) -> bytes:
    decoded_payload = jwt.decode(encoded_token, public_key_path, algorithms=[algorithm])
    return decoded_payload


def hashed_password(password: str) -> bytes:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())


def verify_password(password: str, hashed_password: bytes):
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password)
