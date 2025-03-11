import asyncio
import logging
import pickle
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import redis.asyncio as redis
from insightface.app import FaceAnalysis

from core.config import settings
from core.exceptions import S3Error, FaceNotFoundError, ModelNotFoundError
from schemas.face_meta import TemplateFaceData, FaceMetadata
from services.database.db_template import template_db
from services.file_storage.async_s3_manager import s3_manager, S3Manager

logger = logging.getLogger(__name__)


class AsyncFaceProcessor:
    def __init__(
        self,
        redis_host=settings.redis_config.host,
        redis_port=settings.redis_config.port,
        password=settings.redis_config.password,
    ):
        # Initialize Redis connection
        self.redis = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=password,
            decode_responses=False,
        )

        # Executor for CPU-bound tasks
        self.lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Model-related attributes
        self.model_key = "face_recognition_model"
        self.analyzer = None

    async def initialize_model(self):
        """Initialize or load the face recognition model"""
        # Try to load model from Redis first
        model_bytes = await self.redis.get(self.model_key)

        if model_bytes:
            # Deserialize model from Redis
            try:
                self.analyzer = pickle.loads(model_bytes)
                logger.info("Loaded face recognition model from Redis")
                return
            except Exception as e:
                logger.error(f"Failed to load model from Redis: {e}")

        # If no model in Redis, create a new one
        self.analyzer = FaceAnalysis(name=settings.MODEL_NAME, root=settings.MODEL_PATH)
        self.analyzer.prepare(ctx_id=0)

        # Serialize and store in Redis
        try:
            model_bytes = pickle.dumps(self.analyzer)
            await self.redis.set(self.model_key, model_bytes)
            logger.info("Stored face recognition model in Redis")
        except Exception as e:
            logger.error(f"Failed to store model in Redis: {e}")

    async def reload_model(self):
        """Reload the model from Redis or recreate it"""
        await self.initialize_model()

    async def process_image(self, photo_key: str) -> TemplateFaceData:
        if self.analyzer is None:
            await self.initialize_model()

        image = await s3_manager.download_image(photo_key)
        loop = asyncio.get_event_loop()

        faces = await loop.run_in_executor(self.executor, self.analyzer.get, image)
        if not faces:
            await s3_manager.delete_file(key=photo_key)
            raise FaceNotFoundError

        face = max(
            faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )
        face_data = TemplateFaceData(
            key=photo_key,
            embedding=face.embedding.tolist(),
            metadata=FaceMetadata(
                age=int(face.age),
                gender="Male" if face.gender == 1 else "Female",
                pose=face.pose.tolist(),
                det_score=float(face.det_score),
            ),
        )
        await template_db.add_face(new_face=face_data)

        return face_data


def compute_sim(feat1, feat2):
    try:
        feat1 = feat1.ravel()
        feat2 = feat2.ravel()
        sim = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
        return sim
    except Exception as e:
        logger.error("Failed to compute" + str(e))
        return None


processor = AsyncFaceProcessor()
