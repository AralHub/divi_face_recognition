class FaceRecognitionError(Exception):
    """Base exception for face recognition errors"""

    pass


class FaceNotFoundError(FaceRecognitionError):
    """Raised when no face is detected in the image"""

    pass


class DatabaseError(Exception):
    """Base exception for database errors"""

    pass
