import json
import os
import asyncio
from typing import List, Dict, Any, Optional
from bson import ObjectId


class AsyncFileDB:
    def __init__(self, file_path: str = "data.json"):
        self.file_path = file_path
        self.collections = {}
        self._lock = asyncio.Lock()  # Блокировка для безопасной работы с файлом

    async def _load_data(self):
        """Загрузка данных из JSON-файла"""
        if not os.path.exists(self.file_path):
            return {}

        try:
            async with self._lock:
                with open(self.file_path, "r") as f:
                    return json.load(f)
        except json.JSONDecodeError:
            return {}

    async def _save_data(self, data):
        """Сохранение данных в JSON-файл"""
        async with self._lock:
            with open(self.file_path, "w") as f:
                json.dump(data, f, indent=2)

    async def get_collections_names(self) -> List[str]:
        """Получение списка коллекций"""
        data = await self._load_data()
        return list(data.keys())

    async def get_docs_from_collection(self, collection: str) -> List[dict]:
        """Получение всех документов из коллекции"""
        data = await self._load_data()
        return data.get(collection, [])

    async def add_face_to_collection(self, collection: str, face_data: dict) -> str:
        """Добавление лица в коллекцию"""
        data = await self._load_data()

        if collection not in data:
            data[collection] = []

        # Генерируем ID если его нет
        if "_id" not in face_data:
            face_data["_id"] = str(ObjectId())

        data[collection].append(face_data)
        await self._save_data(data)

        return face_data["_id"]

    async def delete_face(self, collection: str, face_id: str) -> bool:
        """Удаление лица по ID"""
        data = await self._load_data()

        if collection not in data:
            return False

        initial_len = len(data[collection])
        data[collection] = [
            doc for doc in data[collection] if str(doc.get("_id")) != str(face_id)
        ]

        if len(data[collection]) < initial_len:
            await self._save_data(data)
            return True

        return False

    async def delete_person(self, collection: str, person_id: int) -> bool:
        """Удаление всех лиц человека по person_id"""
        data = await self._load_data()

        if collection not in data:
            return False

        initial_len = len(data[collection])
        data[collection] = [
            doc for doc in data[collection] if doc.get("person_id") != person_id
        ]

        if len(data[collection]) < initial_len:
            await self._save_data(data)
            return True

        return False

    async def delete_collection(self, collection: str) -> bool:
        """Удаление коллекции"""
        data = await self._load_data()

        if collection not in data:
            return False

        del data[collection]
        await self._save_data(data)
        return True

    async def get_documents(self):
        """Получение всех документов из базы"""
        data = await self._load_data()
        return data

    async def count_documents(self, collection: str) -> int:
        """Подсчет документов в коллекции"""
        data = await self._load_data()
        return len(data)

    async def get_documents_limit(self, collection: str, limit: int) -> List[dict]:
        """Получение ограниченного количества документов"""
        data = await self._load_data()
        return data[:limit]


# Создаем глобальный экземпляр
db = AsyncFileDB("string.json")
