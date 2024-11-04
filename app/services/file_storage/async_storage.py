import os
from datetime import datetime
from urllib.parse import urljoin

import aiofiles

from core.config import settings


class AsyncFileStorage:
    def __init__(self, base_dir: str, base_url: str):
        self.base_dir = base_dir
        self.base_url = base_url.rstrip("/")  # Remove trailing slash if present

    async def save_file(
        self, file_data: bytes, collection: str, person_id: int
    ) -> tuple[str, str]:
        """
        Save file and return both filepath and URL
        Returns: tuple(filepath, url)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{person_id}_{timestamp}.jpg"

        # Create relative path for the collection/filename
        rel_path = os.path.join(collection, filename)
        # Create absolute filepath for storage
        filepath = os.path.join(self.base_dir, rel_path)

        # Create URL path
        url_path = rel_path.replace(os.path.sep, "/")  # Ensure forward slashes for URLs
        full_url = urljoin(f"{self.base_url}/", url_path)

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        async with aiofiles.open(filepath, "wb") as f:
            await f.write(file_data)

        return filepath, full_url

    @staticmethod
    async def delete_file(self, filepath: str) -> bool:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
            return True
        except Exception:
            return False


storage = AsyncFileStorage(settings.UPLOAD_DIR, settings.MEDIA_URL)
