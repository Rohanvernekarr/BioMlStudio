"""
Simplified analysis endpoints for one-click ML pipeline
"""

import logging
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from app.api.deps import get_current_active_user, get_db
from app.models.user import User
from app.models.dataset import Dataset
from app.schemas.job import JobCreate, JobResponse
from app.tasks.ml_tasks import start_auto_analysis_task

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/auto-analyze/{dataset_id}", response_model=JobResponse)
async def auto_analyze_dataset(
    dataset_id: int,
    target_column: str,
    analysis_type: str = "classification",  # or "regression"
    feature_columns: Optional[str] = None,  # comma-separated or None for auto-detect
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    One-click analysis: Upload CSV, select target, get results.
    
    This endpoint simplifies the ML pipeline to match your specification:
    1. Takes a dataset and target column
    2. Automatically prepares data (clean NaNs, scale, balance if needed)
    3. Trains multiple models (RF + LogisticRegression)
    4. Returns metrics + plots
    
    Args:
        dataset_id: ID of uploaded dataset
        target_column: Column to predict
        analysis_type: "classification" or "regression"
        feature_columns: Specific features (optional, auto-detect if None)
        
    Returns:
        JobResponse: Created analysis job
    """
    # Verify dataset ownership
    dataset = db.query(Dataset).filter(
        Dataset.id == dataset_id,
        Dataset.user_id == current_user.id
    ).first()
    
    if not dataset:
        raise HTTPException(
            status_code=404, 
            detail="Dataset not found or access denied"
        )
    
    # Parse feature columns if provided
    features = None
    if feature_columns:
        features = [col.strip() for col in feature_columns.split(",")]
    
    # Create simplified job configuration
    config = {
        "dataset_id": dataset_id,
        "dataset_path": dataset.file_path,
        "target_column": target_column,
        "feature_columns": features,
        "analysis_type": analysis_type,
        "auto_preprocess": True,  # Enable automatic preprocessing
        "models": ["random_forest", "logistic_regression"],  # Train both
        "generate_plots": True,  # Generate visualizations
        "test_size": 0.2,
        "scale_features": True,
        "handle_imbalance": True if analysis_type == "classification" else False
    }
    
    # Create job using existing job creation logic
    job_data = JobCreate(
        job_type="auto_analysis",
        name=f"Auto Analysis: {dataset.name}",
        description=f"Automated {analysis_type} analysis on {target_column}",
        config=config
    )
    
    # Create job record
    from app.models.job import Job, JobStatus
    from datetime import datetime
    
    db_job = Job(
        user_id=current_user.id,
        job_type=job_data.job_type,
        name=job_data.name,
        description=job_data.description,
        config=job_data.config,
        status=JobStatus.PENDING,
        created_at=datetime.utcnow()
    )
    
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    
    # Start simplified analysis task
    start_auto_analysis_task.delay(job_id=db_job.id, config=config)
    
    # Update status
    db_job.status = JobStatus.QUEUED
    db_job.updated_at = datetime.utcnow()
    db.commit()
    
    logger.info(f"Auto analysis job created: {db_job.id} for dataset {dataset_id}")
    
    return JobResponse.from_orm(db_job)