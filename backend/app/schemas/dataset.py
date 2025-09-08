"""
Dataset-related Pydantic schemas
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class DatasetBase(BaseModel):
    """Base dataset schema"""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    dataset_type: str = Field(..., regex="^(dna|rna|protein|general)$")
    is_public: bool = False


class DatasetCreate(DatasetBase):
    """Schema for dataset creation"""
    pass


class DatasetUpdate(BaseModel):
    """Schema for dataset updates"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    is_public: Optional[bool] = None


class DatasetResponse(DatasetBase):
    """Schema for dataset responses"""
    id: int
    user_id: int
    filename: str
    file_size: int
    file_extension: str
    processing_status: str
    is_validated: bool
    download_count: int = 0
    created_at: datetime
    updated_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None
    stats: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True

    @validator('file_size')
    def format_file_size(cls, v):
        """Convert file size to human readable format"""
        return v


class DatasetListResponse(BaseModel):
    """Schema for paginated dataset list"""
    datasets: List[DatasetResponse]
    total: int
    skip: int
    limit: int


class DatasetPreview(BaseModel):
    """Schema for dataset preview data"""
    dataset_id: int
    preview_data: List[Dict[str, Any]]
    total_rows: int
    columns: Optional[List[str]] = None


class DatasetStats(BaseModel):
    """Schema for dataset statistics"""
    dataset_id: int
    total_rows: int = 0
    total_columns: int = 0
    file_size_bytes: int = 0
    sequence_count: Optional[int] = None
    avg_sequence_length: Optional[float] = None
    min_sequence_length: Optional[int] = None
    max_sequence_length: Optional[int] = None
    gc_content: Optional[float] = None
    n_content: Optional[float] = None
    column_types: Optional[Dict[str, str]] = None
    missing_values: Optional[Dict[str, int]] = None


class DatasetValidation(BaseModel):
    """Schema for dataset validation results"""
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    format_detected: Optional[str] = None
    sequence_type: Optional[str] = None
    encoding: Optional[str] = None
