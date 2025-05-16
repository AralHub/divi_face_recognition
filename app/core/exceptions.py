from fastapi import HTTPException


class FaceRecognitionError(Exception):
    """Base exception for face recognition errors"""

    pass


class DatabaseError(Exception):
    """Base exception for database errors"""

    pass


class CollectionNotFoundError(HTTPException):
    status_code = 404
    detail = "Collection not found"


class FaceNotFoundError(HTTPException):
    status_code = 411
    detail = "no face is detected in the image"


class ModelNotFoundError(Exception):
    status_code = 412
    detail = "model not found"


InvalidDatabase = HTTPException(status_code=413, detail="Invalid Database")

class ImageNoDecodeError(HTTPException):
    status_code = 414
    detail = "Error decoding image"
