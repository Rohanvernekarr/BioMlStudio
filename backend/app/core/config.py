"""
Configuration management using Pydantic Settings
"""

import secrets
from typing import List, Optional, Union

from pydantic import AnyHttpUrl, BaseSettings, EmailStr, validator


class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    # Application
    APP_NAME: str = "BioMLStudio"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    SECRET_KEY: str = secrets.token_urlsafe(32)
    API_VERSION: str = "v1"
    
    # Database
    DATABASE_URL: str = "sqlite:///./bioml.db"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    
    # Redis & Celery
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    
    # MinIO/S3 Storage
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minio"
    MINIO_SECRET_KEY: str = "miniosecret"
    MINIO_BUCKET_NAME: str = "bioml-storage"
    MINIO_SECURE: bool = False
    
    # AWS S3 (alternative)
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_DEFAULT_REGION: str = "us-east-1"
    AWS_S3_BUCKET: Optional[str] = None
    
    # Security
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    PASSWORD_MIN_LENGTH: int = 8
    
    # CORS
    CORS_ORIGINS: List[Union[str, AnyHttpUrl]] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    CORS_ALLOW_CREDENTIALS: bool = True
    
    # File Upload
    MAX_UPLOAD_SIZE_MB: int = 500
    ALLOWED_FILE_EXTENSIONS: List[str] = [
        "csv", "txt", "fasta", "fa", "fastq", "fq", 
        "json", "xlsx", "tsv", "gbk", "gff"
    ]
    UPLOAD_CHUNK_SIZE: int = 8192
    UPLOAD_DIR: str = "uploads"
    
    # ML Settings
    MODEL_STORAGE_PATH: str = "models"
    MAX_TRAINING_TIME_SECONDS: int = 7200  # 2 hours
    DEFAULT_TEST_SIZE: float = 0.2
    MAX_FEATURES: int = 10000
    N_JOBS: int = -1
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    SENTRY_DSN: Optional[str] = None
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 100
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    
    # Email
    SMTP_HOST: Optional[str] = None
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    SMTP_TLS: bool = True
    FROM_EMAIL: EmailStr = "noreply@biomlstudio.com"
    
    # Bioinformatics specific
    MAX_SEQUENCE_LENGTH: int = 50000
    SUPPORTED_SEQUENCE_TYPES: List[str] = ["dna", "rna", "protein"]
    DEFAULT_KMER_SIZE: int = 3
    
    @validator("CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    @validator("DATABASE_URL", pre=True)
    def validate_database_url(cls, v: str) -> str:
        if v and not v.startswith(("postgresql://", "sqlite:///")):
            raise ValueError("Database URL must start with postgresql:// or sqlite:///")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()
