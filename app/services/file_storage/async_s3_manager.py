import aioboto3
import logging
from typing import Any, Optional
from botocore.exceptions import ClientError
from core.config import settings

logger = logging.getLogger(__name__)


class S3Manager:
    def __init__(self):
        self.bucket_name = settings.s3_config.bucket_name
        self.endpoint_url = settings.s3_config.endpoint_url
        self.session = aioboto3.Session()

    async def upload_file(
        self, file_path: str, key: str, content_type: Optional[str] = None
    ):
        """Загрузить файл в S3."""
        try:
            async with self.session.client(
                "s3",
                aws_access_key_id=settings.s3_config.aws_access_key_id,
                aws_secret_access_key=settings.s3_config.aws_secret_access_key,
                region_name=settings.s3_config.region_name,
                endpoint_url=self.endpoint_url,
            ) as s3_client:
                extra_args = {"ContentType": content_type} if content_type else {}
                await s3_client.upload_file(
                    Filename=file_path,
                    Bucket=self.bucket_name,
                    Key=key,
                    ExtraArgs=extra_args,
                )
                logger.info(f"Файл {file_path} загружен как {key}")
        except ClientError as e:
            logger.error(f"Ошибка при загрузке файла {file_path}: {e}")
            raise

    async def download_file(self, key: str, download_path: str):
        """Скачать файл из S3."""
        try:
            async with self.session.client(
                "s3",
                aws_access_key_id=settings.s3_config.aws_access_key_id,
                aws_secret_access_key=settings.s3_config.aws_secret_access_key,
                region_name=settings.s3_config.region_name,
                endpoint_url=self.endpoint_url,
            ) as s3_client:
                await s3_client.download_file(
                    Bucket=self.bucket_name, Key=key, Filename=download_path
                )
                logger.info(f"Файл {key} скачан в {download_path}")
        except ClientError as e:
            logger.error(f"Ошибка при скачивании файла {key}: {e}")
            raise

    async def delete_file(self, key: str):
        """Удалить файл из S3."""
        try:
            async with self.session.client(
                "s3",
                aws_access_key_id=settings.s3_config.aws_access_key_id,
                aws_secret_access_key=settings.s3_config.aws_secret_access_key,
                region_name=settings.s3_config.region_name,
                endpoint_url=self.endpoint_url,
            ) as s3_client:
                await s3_client.delete_object(Bucket=self.bucket_name, Key=key)
                logger.info(f"Файл {key} удален из S3")
        except ClientError as e:
            logger.error(f"Ошибка при удалении файла {key}: {e}")
            raise

    async def get_url_by_key(self, key: str):
        """Получить URL для скачивания файла из S3."""
        try:
            async with self.session.client(
                "s3",
                aws_access_key_id=settings.s3_config.aws_access_key_id,
                aws_secret_access_key=settings.s3_config.aws_secret_access_key,
                region_name=settings.s3_config.region_name,
                endpoint_url=self.endpoint_url,
            ) as s3_client:
                response = await s3_client.generate_presigned_url(
                    ClientMethod="get_object",
                    Params={"Bucket": self.bucket_name, "Key": key},
                    ExpiresIn=3600,  # URL действителен на 1 час
                )
                return response
        except ClientError as e:
            logger.error(f"Error generating {e}")
            raise


s3_client = S3Manager()
