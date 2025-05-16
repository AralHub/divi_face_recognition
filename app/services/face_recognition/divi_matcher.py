import asyncio
import base64
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
from face_sdk_3divi import FacerecService
from face_sdk_3divi.modules.context_template import ContextTemplate
from face_sdk_3divi.modules.dynamic_template_index import DynamicTemplateIndex
from core.config import settings
from core.exceptions import FaceNotFoundError, ModelNotFoundError
from services.face_recognition.divi_service import divi_service
from services.database.mongodb import db

logger = logging.getLogger(__name__)


class DiviMatcher:
    def __init__(self, processor: FacerecService):
        """Инициализация процессора сопоставления лиц 3DiVi"""
        self.use_cuda = settings.USE_CUDA
        self.template_modification = "1000"  # Версия шаблона
        self.max_templates = (
            settings.MAX_TEMPLATES_IN_INDEX
            if hasattr(settings, "MAX_TEMPLATES_IN_INDEX")
            else 1000
        )
        self.executor = ThreadPoolExecutor(max_workers=settings.WORKER_POOL_SIZE)
        self.processor = processor
        self.service = None  # Будет использоваться из divi_service
        self.matcher_module = None
        self.verification_module = None
        self.template_index = None
        self.template_name = None
        self.uuids = []

    async def initialize(self):
        """Инициализация компонентов 3DiVi Face SDK для сопоставления"""
        try:
            loop = asyncio.get_event_loop()

            # Используем глобальный сервис
            self.service = divi_service.service

            # Создаем модуль сопоставления
            self.matcher_module = await loop.run_in_executor(
                self.executor, self._create_matcher_module
            )

            # Создаем индекс шаблонов
            self.template_index = await loop.run_in_executor(
                self.executor, self._create_template_index
            )

            logger.info("3DiVi Matcher инициализирован успешно")
        except Exception as e:
            logger.error(f"Ошибка инициализации 3DiVi Matcher: {e}")
            raise ModelNotFoundError("Не удалось инициализировать 3DiVi Matcher")

    def _create_template_index(self):
        """Создание индекса шаблонов"""
        # Создаем контекст конфигурации вместо прямой передачи словаря
        config = self.service.create_context(
            {
                "capacity": self.max_templates,
                "model_version": f"{self.template_modification}_1",
                "max_license_count": self.max_templates,
            }
        )

        # Теперь передаем контекст, а не словарь
        return self.service.create_dynamic_template_index(config)

    def _create_matcher_module(self):
        """Создание модуля сопоставления"""
        return self.service.create_processing_block(
            {
                "unit_type": "MATCHER_MODULE",
                "modification": self.template_modification,
                "use_cuda": self.use_cuda,
            }
        )

    async def reset_index(self, uuids: List[str]):
        """Очистка индекса шаблонов"""
        if not self.template_index:
            await self.initialize()

        loop = asyncio.get_event_loop()

        if uuids:
            # Удаляем все шаблоны из индекса
            await loop.run_in_executor(
                self.executor, lambda: self.template_index.remove(uuids)
            )

        # Сбрасываем переменные
        self.template_name = None
        self.uuids = []
        return True

    async def add_templates(self, templates: List[str], uuids: List[str]) -> bool:
        """Добавление списка шаблонов в индекс по base64 строкам"""
        if not self.template_index:
            await self.initialize()

        loop = asyncio.get_event_loop()

        for template_base64, uuid in zip(templates, uuids):
            # Декодирование шаблона из base64
            template_bytes = base64.b64decode(template_base64)

            # Загружаем шаблон
            template = await loop.run_in_executor(
                self.executor,
                lambda: self.service.load_context_template(BytesIO(template_bytes)),
            )

            # Добавляем шаблон в индекс
            await loop.run_in_executor(
                self.executor, lambda: self.template_index.add(template, uuid)
            )

        # Сохраняем UUID для возможной последующей очистки
        self.uuids.extend(uuids)

        return True

    async def search_face(
        self,
        template_base64: str,
        top_n: int = 1,
    ) -> List[Dict[str, Any]]:
        """Поиск лица в базе шаблонов"""
        if not self.template_index:
            await self.initialize()

        # Проверяем размер индекса
        template_count = await self.get_templates_count()
        if template_count == 0:
            return []

        loop = asyncio.get_event_loop()

        # Декодирование шаблона из base64
        template_bytes = base64.b64decode(template_base64)

        # Загружаем шаблон
        template = await loop.run_in_executor(
            self.executor,
            lambda: self.service.load_context_template(BytesIO(template_bytes)),
        )

        # Создаем контекст для поиска
        matcher_data = await loop.run_in_executor(
            self.executor, lambda: self.service.create_context()
        )

        # Заполняем контекст
        await loop.run_in_executor(
            self.executor,
            lambda: self._setup_matcher_context(matcher_data, template, top_n),
        )

        # Выполняем поиск
        await loop.run_in_executor(
            self.executor, lambda: self.matcher_module(matcher_data)
        )

        # Получаем результаты
        results = await loop.run_in_executor(
            self.executor, lambda: self._extract_matcher_results(matcher_data)
        )

        return results

    def _setup_matcher_context(self, ctx, template, top_n):
        """Заполнение контекста для поиска"""
        # Добавляем индекс шаблонов в контекст
        ctx["template_index"] = self.template_index

        # Добавляем шаблон запроса
        ctx["queries"] = template  # Передаем список шаблонов

    def _extract_matcher_results(self, ctx) -> List[Dict[str, Any]]:
        """Извлечение результатов поиска"""
        results = []
        # Проверяем, содержит ли контекст результаты
        if ctx.contains("results"):

            # Получаем массив результатов
            result_array = ctx["results"]

            # Обрабатываем каждый результат
            for result in result_array:
                # Добавляем данные в списко результатов
                result_dict = {
                    "index": int(result["index"].get_long()),
                    "uuid": result["uuid"].get_string(),
                    "distance": float(result["distance"].get_double()),
                    "score": float(result["score"].get_double()),
                    "far": float(result["far"].get_double()),
                    "frr": float(result["frr"].get_double()),
                }
                results.append(result_dict)

        return results

    async def remove_template(self, uuids: List[str]) -> bool:
        """Удаление шаблона из индекса по uuid"""
        if not self.template_index:
            await self.initialize()

        loop = asyncio.get_event_loop()

        # Удаляем шаблон
        await loop.run_in_executor(
            self.executor, lambda: self.template_index.remove(uuids)
        )

        return True

    async def get_templates_count(self) -> int:
        """Получение количества шаблонов в индексе"""
        if not self.template_index:
            await self.initialize()

        loop = asyncio.get_event_loop()

        # Получаем размер индекса
        size = await loop.run_in_executor(
            self.executor, lambda: self.template_index.size()
        )

        return size

    async def get_templates_count(self) -> int:
        """Получение UUID шаблонов в индексе"""
        if not self.template_index:
            await self.initialize()

        return self.template_index.size()

    async def get_template_name(self) -> str:
        return self.template_index.get_method_name()


# Создаем глобальный экземпляр маттчера
matcher = DiviMatcher(processor=divi_service.service)
