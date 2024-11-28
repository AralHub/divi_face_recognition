import asyncio
import logging
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple
from urllib.parse import urljoin

import cv2
import numpy as np
import redis.asyncio as redis
from insightface.app import FaceAnalysis

from core.config import settings

logger = logging.getLogger(__name__)


class AsyncFaceProcessor:
    def __init__(
        self, redis_host="localhost", redis_port=6379, redis_password="redis123"
    ):
        # Initialize Redis connection
        self.redis = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            decode_responses=False,
        )

        # Executor for CPU-bound tasks
        self.executor = ThreadPoolExecutor(max_workers=settings.WORKER_POOL_SIZE)

        # Lock for thread-safe operations
        self.lock = asyncio.Lock()

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

    async def process_image(
        self, image_data: bytes
    ) -> Optional[Tuple[np.ndarray, dict]]:
        # Ensure model is initialized
        if self.analyzer is None:
            await self.initialize_model()

        loop = asyncio.get_event_loop()
        image_array = await loop.run_in_executor(
            self.executor, lambda: np.frombuffer(image_data, np.uint8)
        )
        image = await loop.run_in_executor(
            self.executor, lambda: cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        )

        if image is None:
            return None

        # Use the model for face detection
        faces = await loop.run_in_executor(self.executor, self.analyzer.get, image)
        if not faces:
            return None

        # Select the largest face
        face = max(
            faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )

        return face.embedding, {
            "age": int(face.age),
            "gender": "Male" if face.gender == 1 else "Female",
            "pose": face.pose.tolist(),
            "det_score": float(face.det_score),
        }

    async def background_image(
        self, snap_image: bytes, background_image: bytes, filename: str
    ):
        loop = asyncio.get_event_loop()
        snap_image_array = await loop.run_in_executor(
            self.executor, lambda: np.frombuffer(snap_image, np.uint8)
        )
        background_image_array = await loop.run_in_executor(
            self.executor, lambda: np.frombuffer(background_image, np.uint8)
        )

        snap_image = await loop.run_in_executor(
            self.executor, lambda: cv2.imdecode(snap_image_array, cv2.IMREAD_COLOR)
        )
        background_image = await loop.run_in_executor(
            self.executor,
            lambda: cv2.imdecode(background_image_array, cv2.IMREAD_COLOR),
        )

        if background_image is None or snap_image is None:
            return None

        faces = await loop.run_in_executor(self.executor, self.analyzer.get, snap_image)
        if not faces:
            return None

        face = max(
            faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )

        faces_back = await loop.run_in_executor(
            self.executor, self.analyzer.get, background_image
        )

        rel_path = os.path.join("background_images/", filename)
        filepath = os.path.join(settings.UPLOAD_DIR, rel_path)
        url_path = rel_path.replace(os.path.sep, "/")
        full_url = urljoin(f"{settings.MEDIA_URL}/", url_path)

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if not faces_back:
            return None
        for data in faces_back:
            if compute_sim(data.embedding, face.embedding) > 0.8:
                x1, y1, x2, y2 = map(int, data.bbox)
                cv2.rectangle(background_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imwrite(filepath, background_image)
        return {"background_image_path": filepath, "background_image_url": full_url}


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
