import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from core.config import settings


class AsyncFaceProcessor:
    def __init__(self):
        self.analyzer = FaceAnalysis(name=settings.MODEL_NAME)
        self.analyzer.prepare(ctx_id=0)
        self.executor = ThreadPoolExecutor(max_workers=settings.WORKER_POOL_SIZE)

    async def process_image(
        self, image_data: bytes
    ) -> Optional[Tuple[np.ndarray, dict]]:
        # Конвертируем bytes в numpy array асинхронно
        loop = asyncio.get_event_loop()
        image_array = await loop.run_in_executor(
            self.executor, lambda: np.frombuffer(image_data, np.uint8)
        )

        # Декодируем изображение из массива байтов в формат, понятный OpenCV
        image = await loop.run_in_executor(
            self.executor, lambda: cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        )

        if image is None:
            # Логирование, если изображение не может быть загружено
            return None

        # Получаем эмбеддинги через thread pool
        faces = await loop.run_in_executor(self.executor, self.analyzer.get, image)
        if not faces:
            return None

        face = max(
            faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )

        return face.embedding, {
            "age": int(face.age),
            "gender": "Male" if face.gender == 1 else "Female",
            "pose": face.pose.tolist(),
        }


processor = AsyncFaceProcessor()
