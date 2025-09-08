"""
Storage service for managing files and model artifacts
"""

import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

from minio import Minio
from minio.error import S3Error

from app.core.config import settings

logger = logging.getLogger(__name__)


class StorageService:
    """Service for file and artifact storage"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize MinIO client
        self.client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE
        )
        
        # Ensure bucket exists
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self) -> None:
        """Ensure the storage bucket exists"""
        try:
            if not self.client.bucket_exists(settings.MINIO_BUCKET_NAME):
                self.client.make_bucket(settings.MINIO_BUCKET_NAME)
                self.logger.info(f"Created bucket: {settings.MINIO_BUCKET_NAME}")
        except S3Error as e:
            self.logger.error(f"Error creating bucket: {e}")
            raise
    
    async def upload_file(
        self,
        file_path: str,
        object_name: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Upload file to storage.
        
        Args:
            file_path: Local file path
            object_name: Object name in storage
            metadata: Optional metadata
            
        Returns:
            str: Storage object path
        """
        try:
            self.client.fput_object(
                settings.MINIO_BUCKET_NAME,
                object_name,
                file_path,
                metadata=metadata
            )
            
            self.logger.info(f"File uploaded: {object_name}")
            return f"{settings.MINIO_BUCKET_NAME}/{object_name}"
            
        except S3Error as e:
            self.logger.error(f"Error uploading file {object_name}: {e}")
            raise
    
    async def download_file(
        self,
        object_name: str,
        file_path: str
    ) -> bool:
        """
        Download file from storage.
        
        Args:
            object_name: Object name in storage
            file_path: Local file path to save
            
        Returns:
            bool: True if downloaded successfully
        """
        try:
            self.client.fget_object(
                settings.MINIO_BUCKET_NAME,
                object_name,
                file_path
            )
            
            self.logger.info(f"File downloaded: {object_name} -> {file_path}")
            return True
            
        except S3Error as e:
            self.logger.error(f"Error downloading file {object_name}: {e}")
            return False
    
    async def get_file_stream(self, object_name: str) -> BytesIO:
        """
        Get file as stream.
        
        Args:
            object_name: Object name in storage
            
        Returns:
            BytesIO: File stream
        """
        try:
            response = self.client.get_object(
                settings.MINIO_BUCKET_NAME,
                object_name
            )
            
            data = response.read()
            return BytesIO(data)
            
        except S3Error as e:
            self.logger.error(f"Error getting file stream {object_name}: {e}")
            raise
    
    async def delete_file(self, object_name: str) -> bool:
        """
        Delete file from storage.
        
        Args:
            object_name: Object name to delete
            
        Returns:
            bool: True if deleted successfully
        """
        try:
            self.client.remove_object(
                settings.MINIO_BUCKET_NAME,
                object_name
            )
            
            self.logger.info(f"File deleted: {object_name}")
            return True
            
        except S3Error as e:
            self.logger.error(f"Error deleting file {object_name}: {e}")
            return False
    
    async def list_files(self, prefix: str = "") -> list:
        """
        List files with optional prefix.
        
        Args:
            prefix: Object name prefix
            
        Returns:
            list: List of object information
        """
        try:
            objects = self.client.list_objects(
                settings.MINIO_BUCKET_NAME,
                prefix=prefix,
                recursive=True
            )
            
            return [
                {
                    'name': obj.object_name,
                    'size': obj.size,
                    'last_modified': obj.last_modified,
                    'etag': obj.etag
                }
                for obj in objects
            ]
            
        except S3Error as e:
            self.logger.error(f"Error listing files with prefix {prefix}: {e}")
            return []
    
    async def get_file_info(self, object_name: str) -> Optional[Dict[str, Any]]:
        """
        Get file information.
        
        Args:
            object_name: Object name
            
        Returns:
            Dict: File information or None if not found
        """
        try:
            stat = self.client.stat_object(
                settings.MINIO_BUCKET_NAME,
                object_name
            )
            
            return {
                'name': object_name,
                'size': stat.size,
                'last_modified': stat.last_modified,
                'etag': stat.etag,
                'content_type': stat.content_type,
                'metadata': stat.metadata
            }
            
        except S3Error as e:
            if e.code == 'NoSuchKey':
                return None
            self.logger.error(f"Error getting file info {object_name}: {e}")
            raise
    
    async def get_model_file(
        self,
        model_path: str,
        format: str = "joblib"
    ) -> BytesIO:
        """
        Get model file in specified format.
        
        Args:
            model_path: Path to model file
            format: Export format (joblib, pickle, onnx)
            
        Returns:
            BytesIO: Model file stream
        """
        # For now, just return the original file
        # In future, could implement format conversion
        return await self.get_file_stream(model_path)
    
    async def delete_model_files(self, model_path: str) -> bool:
        """
        Delete all files associated with a model.
        
        Args:
            model_path: Base model path
            
        Returns:
            bool: True if deleted successfully
        """
        try:
            # List all files with model path as prefix
            model_dir = str(Path(model_path).parent)
            files = await self.list_files(prefix=model_dir)
            
            # Delete all associated files
            for file_info in files:
                await self.delete_file(file_info['name'])
            
            self.logger.info(f"Model files deleted: {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting model files {model_path}: {e}")
            return False
