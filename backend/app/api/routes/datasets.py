"""
Dataset management endpoints for uploading, processing, and managing biological datasets
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc

from app.api.deps import (
    get_current_active_user, get_db, CommonQueryParams, 
    validate_file_upload
)
from app.core.config import settings
from app.core.exceptions import BioMLException
from app.models.dataset import Dataset
from app.models.user import User
from app.schemas.dataset import (
    DatasetResponse, DatasetUpdate, DatasetListResponse,
    DatasetPreview, DatasetStats
)
from app.services.dataset_service import DatasetService
from app.services.storage_service import StorageService
from app.utils.bioinformatics import validate_biological_file
from app.utils.file_handlers import get_file_info, generate_unique_filename

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/upload", response_model=DatasetResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: Optional[str] = Form(None),
    dataset_type: str = Form("general"),
    is_public: bool = Form(False),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Upload a biological dataset file.
    
    Args:
        file: Uploaded file
        name: Dataset name
        description: Dataset description
        dataset_type: Type of dataset (dna, protein, rna, general)
        is_public: Whether dataset should be public
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        DatasetResponse: Created dataset information
    """
    # Validate file
    file_info = get_file_info(file.filename)
    await validate_file_upload(file.size, file_info['extension'])
    
    # Validate biological file format if needed
    if dataset_type in ['dna', 'protein', 'rna']:
        is_valid = await validate_biological_file(file, dataset_type)
        if not is_valid:
            raise BioMLException(
                status_code=400,
                detail=f"Invalid {dataset_type} file format",
                error_code="INVALID_BIOLOGICAL_FORMAT"
            )
    
    try:
        # Generate unique filename
        unique_filename = generate_unique_filename(
            file.filename, 
            current_user.id
        )
        
        # Create upload directory if it doesn't exist
        upload_dir = Path(settings.UPLOAD_DIR) / str(current_user.id)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / unique_filename
        
        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Create dataset record
        dataset_service = DatasetService()
        
        # Analyze dataset
        stats = await dataset_service.analyze_dataset(file_path, dataset_type)
        
        db_dataset = Dataset(
            user_id=current_user.id,
            name=name,
            description=description,
            dataset_type=dataset_type,
            file_path=str(file_path),
            filename=file.filename,
            file_size=file.size,
            file_extension=file_info['extension'],
            stats=stats,
            is_public=is_public,
            created_at=datetime.utcnow()
        )
        
        db.add(db_dataset)
        db.commit()
        db.refresh(db_dataset)
        
        logger.info(f"Dataset uploaded: {db_dataset.id} by user {current_user.id}")
        
        return DatasetResponse.from_orm(db_dataset)
        
    except Exception as e:
        logger.error(f"Error uploading dataset: {e}")
        # Clean up file if database save failed
        if 'file_path' in locals() and file_path.exists():
            file_path.unlink()
        
        raise BioMLException(
            status_code=500,
            detail="Failed to upload dataset",
            error_code="DATASET_UPLOAD_FAILED"
        )


@router.get("/", response_model=DatasetListResponse)
async def list_datasets(
    commons: CommonQueryParams = Depends(CommonQueryParams),
    dataset_type: Optional[str] = None,
    is_public: Optional[bool] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    List user's datasets with filtering and pagination.
    
    Args:
        commons: Common query parameters
        dataset_type: Filter by dataset type
        is_public: Filter by public/private datasets
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        DatasetListResponse: Paginated dataset list
    """
    query = db.query(Dataset).filter(
        (Dataset.user_id == current_user.id) | (Dataset.is_public == True)
    )
    
    # Apply filters
    if dataset_type:
        query = query.filter(Dataset.dataset_type == dataset_type)
    if is_public is not None:
        query = query.filter(Dataset.is_public == is_public)
    
    # Apply sorting
    if commons.sort_order == "asc":
        query = query.order_by(asc(getattr(Dataset, commons.sort_by, Dataset.created_at)))
    else:
        query = query.order_by(desc(getattr(Dataset, commons.sort_by, Dataset.created_at)))
    
    # Get total count
    total = query.count()
    
    # Apply pagination
    datasets = query.offset(commons.skip).limit(commons.limit).all()
    
    return DatasetListResponse(
        datasets=[DatasetResponse.from_orm(dataset) for dataset in datasets],
        total=total,
        skip=commons.skip,
        limit=commons.limit
    )


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Get specific dataset details.
    
    Args:
        dataset_id: Dataset ID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        DatasetResponse: Dataset details
    """
    dataset = db.query(Dataset).filter(
        Dataset.id == dataset_id,
        (Dataset.user_id == current_user.id) | (Dataset.is_public == True)
    ).first()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    return DatasetResponse.from_orm(dataset)


@router.get("/{dataset_id}/preview", response_model=DatasetPreview)
async def preview_dataset(
    dataset_id: int,
    rows: int = 10,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Preview dataset content (first few rows/sequences).
    
    Args:
        dataset_id: Dataset ID
        rows: Number of rows/sequences to preview
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        DatasetPreview: Dataset preview data
    """
    dataset = db.query(Dataset).filter(
        Dataset.id == dataset_id,
        (Dataset.user_id == current_user.id) | (Dataset.is_public == True)
    ).first()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    dataset_service = DatasetService()
    preview_data = await dataset_service.preview_dataset(
        dataset.file_path,
        dataset.dataset_type,
        rows=rows
    )
    
    return DatasetPreview(
        dataset_id=dataset_id,
        preview_data=preview_data,
        total_rows=dataset.stats.get('total_rows', 0) if dataset.stats else 0
    )


@router.get("/{dataset_id}/stats", response_model=DatasetStats)
async def get_dataset_stats(
    dataset_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Get detailed dataset statistics.
    
    Args:
        dataset_id: Dataset ID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        DatasetStats: Dataset statistics
    """
    dataset = db.query(Dataset).filter(
        Dataset.id == dataset_id,
        (Dataset.user_id == current_user.id) | (Dataset.is_public == True)
    ).first()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    return DatasetStats(
        dataset_id=dataset_id,
        **dataset.stats if dataset.stats else {}
    )


@router.get("/{dataset_id}/download")
async def download_dataset(
    dataset_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> StreamingResponse:
    """
    Download dataset file.
    
    Args:
        dataset_id: Dataset ID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        StreamingResponse: Dataset file download
    """
    dataset = db.query(Dataset).filter(
        Dataset.id == dataset_id,
        (Dataset.user_id == current_user.id) | (Dataset.is_public == True)
    ).first()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    file_path = Path(dataset.file_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset file not found"
        )
    
    async def file_generator():
        async with aiofiles.open(file_path, 'rb') as f:
            while chunk := await f.read(settings.UPLOAD_CHUNK_SIZE):
                yield chunk
    
    return StreamingResponse(
        file_generator(),
        media_type='application/octet-stream',
        headers={
            "Content-Disposition": f"attachment; filename={dataset.filename}"
        }
    )


@router.put("/{dataset_id}", response_model=DatasetResponse)
async def update_dataset(
    dataset_id: int,
    dataset_update: DatasetUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Update dataset metadata.
    
    Args:
        dataset_id: Dataset ID
        dataset_update: Dataset update data
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        DatasetResponse: Updated dataset information
    """
    dataset = db.query(Dataset).filter(
        Dataset.id == dataset_id,
        Dataset.user_id == current_user.id
    ).first()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    # Apply updates
    for field, value in dataset_update.dict(exclude_unset=True).items():
        setattr(dataset, field, value)
    
    dataset.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(dataset)
    
    logger.info(f"Dataset updated: {dataset.id}")
    
    return DatasetResponse.from_orm(dataset)


@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Dict[str, str]:
    """
    Delete dataset and its file.
    
    Args:
        dataset_id: Dataset ID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Dict: Success message
    """
    dataset = db.query(Dataset).filter(
        Dataset.id == dataset_id,
        Dataset.user_id == current_user.id
    ).first()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    # Delete file
    file_path = Path(dataset.file_path)
    if file_path.exists():
        file_path.unlink()
    
    # Delete dataset record
    db.delete(dataset)
    db.commit()
    
    logger.info(f"Dataset deleted: {dataset_id}")
    
    return {"message": "Dataset deleted successfully"}


@router.post("/{dataset_id}/validate")
async def validate_dataset(
    dataset_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Validate dataset format and content.
    
    Args:
        dataset_id: Dataset ID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Dict: Validation results
    """
    dataset = db.query(Dataset).filter(
        Dataset.id == dataset_id,
        (Dataset.user_id == current_user.id) | (Dataset.is_public == True)
    ).first()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    dataset_service = DatasetService()
    validation_results = await dataset_service.validate_dataset(
        dataset.file_path,
        dataset.dataset_type
    )
    
    return {
        "dataset_id": dataset_id,
        "validation_results": validation_results,
        "timestamp": datetime.utcnow()
    }
