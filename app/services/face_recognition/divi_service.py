import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from face_sdk_3divi import FacerecService

from core.config import settings
from core.exceptions import ModelNotFoundError

logger = logging.getLogger(__name__)


class DiviService:
    def __init__(self):
        """Инициализация сервиса 3DiVi Face SDK"""
        self.sdk_path = settings.DIVI_SDK_PATH
        self.use_cuda = settings.USE_CUDA
        self.executor = ThreadPoolExecutor(max_workers=settings.WORKER_POOL_SIZE)
        self._service = None

    @property
    def service(self):
        """Получение экземпляра сервиса, с ленивой инициализацией"""
        if self._service is None:
            try:
                self._service = self._create_service()
                logger.info("3DiVi Face SDK сервис успешно инициализирован")
            except Exception as e:
                logger.error(f"Ошибка инициализации 3DiVi Face SDK сервиса: {e}")
                raise ModelNotFoundError(
                    "Не удалось инициализировать 3DiVi Face SDK сервис"
                )
        return self._service

    def _create_service(self):
        """Создание сервиса 3DiVi SDK"""
        sdk_conf_dir = os.path.join(self.sdk_path, "conf", "facerec")
        sdk_dll_path = os.path.join(
            self.sdk_path,
            "lib" if os.name != "nt" else "bin",
            "libfacerec.so" if os.name != "nt" else "facerec.dll",
        )

        return FacerecService.create_service(
            sdk_dll_path, sdk_conf_dir, f"{self.sdk_path}/license"
        )


# Создаем глобальный экземпляр сервиса
divi_service = DiviService()
