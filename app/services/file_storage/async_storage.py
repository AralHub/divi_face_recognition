import os
from datetime import datetime

import aiofiles

from core.config import settings


class AsyncFileStorage:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    async def save_file(self, file_data: bytes, collection: str, person_id: int) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{person_id}_{timestamp}.jpg"
        filepath = os.path.join(self.base_dir, collection, filename)

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        async with aiofiles.open(filepath, "wb") as f:
            await f.write(file_data)

        return filepath

    async def delete_file(self, filepath: str) -> bool:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
            return True
        except Exception:
            return False


storage = AsyncFileStorage(settings.UPLOAD_DIR)
