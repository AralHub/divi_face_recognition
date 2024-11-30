from fastapi import HTTPException


class FaceRecognitionError(Exception):
    """Base exception for face recognition errors"""

    pass


class DatabaseError(Exception):
    """Base exception for database errors"""

    pass


S3Error = HTTPException(status_code=511, detail="S3 server error")
FaceNotFoundError = HTTPException(
    status_code=411, detail="no face is detected in the image"
)
ModelNotFoundError = HTTPException(status_code=412, detail="model not found")

InvalidDatabase = HTTPException(status_code=413, detail="Invalid Database")

ImageNoDecodeError = HTTPException(status_code=414, detail="Error decoding image")
