import asyncio
import base64
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Dict, List, Tuple, Optional

import numpy as np
from face_sdk_3divi import FacerecService
from face_sdk_3divi.modules.context_template import ContextTemplate

from core.config import settings
from core.exceptions import FaceNotFoundError, ModelNotFoundError
from schemas.face_meta import FaceMeta
from services.face_recognition.divi_service import divi_service

logger = logging.getLogger(__name__)


class DiviFaceProcessor:
    def __init__(self):
        self.use_cuda = settings.USE_CUDA
        self.template_modification = "1000"  # Версия шаблона, можно вынести в настройки
        self.executor = ThreadPoolExecutor(max_workers=settings.WORKER_POOL_SIZE)
        self.service = None  # Будет использоваться из divi_service
        self.detector = None
        self.fitter = None
        self.template_extractor = None
        self.quality_estimator = None
        self.age_estimator = None
        self.gender_estimator = None

    async def initialize(self):
        """Инициализация компонентов 3DiVi Face SDK"""
        try:
            loop = asyncio.get_event_loop()
            self.service = divi_service.service
            self.detector = await loop.run_in_executor(
                self.executor, self._create_detector
            )
            self.fitter = await loop.run_in_executor(self.executor, self._create_fitter)
            self.template_extractor = await loop.run_in_executor(
                self.executor, self._create_template_extractor
            )
            self.quality_estimator = await loop.run_in_executor(
                self.executor, self._create_quality_estimator
            )
            self.age_estimator = await loop.run_in_executor(
                self.executor, self._create_age_estimator
            )
            self.gender_estimator = await loop.run_in_executor(
                self.executor, self._create_gender_estimator
            )
            logger.info("3DiVi Face SDK инициализирован успешно")

        except Exception as e:
            logger.error(f"Ошибка инициализации 3DiVi Face SDK: {e}")
            raise ModelNotFoundError("Не удалось инициализировать 3DiVi Face SDK")

    def _create_detector(self):
        """Создание детектора лиц"""
        return self.service.create_processing_block(
            {
                "unit_type": "FACE_DETECTOR",
                "modification": "ssyv_light",
                "use_cuda": self.use_cuda,
            }
        )

    def _create_fitter(self):
        """Создание модуля определения ключевых точек лица"""
        return self.service.create_processing_block(
            {
                "unit_type": "FACE_FITTER",
                "modification": "fda",
                "use_cuda": self.use_cuda,
            }
        )

    def _create_template_extractor(self):
        """Создание экстрактора шаблонов лиц"""
        return self.service.create_processing_block(
            {
                "unit_type": "FACE_TEMPLATE_EXTRACTOR",
                "modification": self.template_modification,
                "use_cuda": self.use_cuda,
            }
        )

    def _create_quality_estimator(self):
        """Создание модуля оценки качества лица"""
        return self.service.create_processing_block(
            {
                "unit_type": "QUALITY_ASSESSMENT_ESTIMATOR",
                "modification": "assessment",
                "version": 2,
                "use_cuda": self.use_cuda,
                "enable_check_eye_distance": True,
                "enable_check_rotation": True,
            }
        )

    def _create_age_estimator(self):
        """Создание модуля определения возраста"""
        return self.service.create_processing_block(
            {
                "unit_type": "AGE_ESTIMATOR",
                "modification": "heavy",  # heavy имеет лучшую точность
                "use_cuda": self.use_cuda,
            }
        )

    def _create_gender_estimator(self):
        """Создание модуля определения пола"""
        return self.service.create_processing_block(
            {
                "unit_type": "GENDER_ESTIMATOR",
                "modification": "heavy",  # heavy имеет лучшую точность
                "use_cuda": self.use_cuda,
            }
        )

    async def process_image_bytes(
        self, image_bytes: bytes
    ) -> Tuple[ContextTemplate, FaceMeta]:
        """Обработка изображения из байтов"""
        if not self.service:
            await self.initialize()

        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            self.executor,
            lambda: self.service.create_context_from_encoded_image(image_bytes),
        )

        # Детектирование лица
        await loop.run_in_executor(self.executor, self.detector, data)
        if not data.contains("objects") or len(data["objects"]) == 0:
            raise FaceNotFoundError("Лицо не обнаружено в изображении")

        # Определение ключевых точек
        await loop.run_in_executor(self.executor, self.fitter, data)

        # Оценка качества
        await loop.run_in_executor(self.executor, self.quality_estimator, data)

        # Определение возраста
        await loop.run_in_executor(self.executor, self.age_estimator, data)

        # Определение пола
        await loop.run_in_executor(self.executor, self.gender_estimator, data)

        # Извлечение шаблона
        await loop.run_in_executor(self.executor, self.template_extractor, data)

        # Извлечение метаданных
        quality = data["objects"][0]["quality"]

        face_meta = FaceMeta(
            quality_score=(
                float(quality["total_score"].get_double())
                if "total_score" in quality
                else None
            ),
            rotation=(
                float(quality["max_rotation_deviation"].get_long())
                if "max_rotation_deviation" in quality
                else None
            ),
            eyes_distance=(
                int(quality["eyes_distance"].get_long())
                if "eyes_distance" in quality
                else None
            ),
            age=(
                int(data["objects"][0]["age"].get_long())
                if data["objects"][0].contains("age")
                else None
            ),
            gender=(
                data["objects"][0]["gender"].get_string()
                if data["objects"][0].contains("gender")
                else None
            ),
            emotions=None,
        )

        # Возвращаем шаблон и метаданные
        template = data["objects"][0]["face_template"]["template"].get_value()
        return template, face_meta

    async def process_image(self, photo_path: str) -> Tuple[str, FaceMeta]:
        """Обработка изображения по ключу из хранилища"""

        with open(photo_path, "rb") as image_file:
            image_bytes = image_file.read()

        template, face_meta = await self.process_image_bytes(image_bytes)

        # Сериализуем шаблон в base64 для хранения
        buffer = BytesIO()
        template.save(buffer)
        buffer.seek(0)
        template_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        return template_base64, face_meta


processor = DiviFaceProcessor()
