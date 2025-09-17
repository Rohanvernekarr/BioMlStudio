"""
Machine learning background tasks
"""

import logging
import traceback
from datetime import datetime
from typing import Any, Dict

from celery import current_task
from sqlalchemy.orm import sessionmaker

from app.core.celery_app import celery_app
from app.core.database import engine
from app.models.job import Job, JobStatus
from app.models.ml_model import MLModel, ModelType, ModelFramework
from app.services.ml_service import MLService
from app.services.job_service import JobService
from app.utils.logger import get_task_logger

logger = get_task_logger(__name__)
SessionLocal = sessionmaker(bind=engine)


@celery_app.task(bind=True, name='biomlstudio.ml_tasks.start_training_task')
def start_training_task(
    self, 
    job_id: int, 
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Start ML model training task.
    
    Args:
        job_id: Job ID
        config: Training configuration
        
    Returns:
        Dict: Task result
    """
    task_id = self.request.id
    logger.info(f"Starting training task {task_id} for job {job_id}")
    
    job_service = JobService()
    ml_service = MLService()
    
    try:
        # Update job status to running
        job_service.update_job_status(job_id, JobStatus.RUNNING)
        job_service.add_job_log(job_id, f"Training task started: {task_id}")
        
        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={'current': 0, 'total': 100, 'status': 'Initializing training...'}
        )
        
        # Get training configuration
        model_type = config.get('model_type', 'classification')
        dataset_path = config.get('dataset_path')
        
        if not dataset_path:
            raise ValueError("Dataset path is required")
        
        # Update progress
        job_service.update_job_progress(job_id, 10.0, "Loading dataset")
        self.update_state(
            state='PROGRESS', 
            meta={'current': 10, 'total': 100, 'status': 'Loading dataset...'}
        )
        
        # Start training based on model type
        if model_type == 'classification':
            result = ml_service.train_classification_model(
                job_id=job_id,
                dataset_path=dataset_path,
                config=config
            )
        elif model_type == 'regression':
            result = ml_service.train_regression_model(
                job_id=job_id,
                dataset_path=dataset_path,
                config=config
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Update progress
        job_service.update_job_progress(job_id, 80.0, "Saving model")
        self.update_state(
            state='PROGRESS',
            meta={'current': 80, 'total': 100, 'status': 'Saving model...'}
        )
        
        # Create ML model record
        with SessionLocal() as db:
            job = db.query(Job).filter(Job.id == job_id).first()
            
            ml_model = MLModel(
                user_id=job.user_id,
                name=config.get('model_name', f"Model_Job_{job_id}"),
                description=config.get('model_description'),
                model_type=ModelType(model_type),
                framework=ModelFramework.SCIKIT_LEARN,
                algorithm=result['algorithm'],
                artifact_path=result['model_path'],
                hyperparameters=config.get('hyperparameters', {}),
                metrics=result['metrics'],
                feature_importance=result.get('feature_importance'),
                training_samples_count=result['training_samples'],
                validation_samples_count=result['test_samples'],
                created_at=datetime.utcnow()
            )
            
            db.add(ml_model)
            db.commit()
            db.refresh(ml_model)
            
            # Link model to job
            job.model_id = ml_model.id
            db.commit()
        
        # Complete job
        job_service.update_job_status(
            job_id, 
            JobStatus.COMPLETED, 
            metrics=result['metrics']
        )
        job_service.update_job_progress(job_id, 100.0, "Training completed")
        job_service.add_job_log(job_id, "Training completed successfully")
        
        logger.info(f"Training task completed for job {job_id}")
        
        return {
            'status': 'completed',
            'job_id': job_id,
            'model_id': ml_model.id,
            'metrics': result['metrics'],
            'task_id': task_id
        }
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        
        logger.error(f"Training task failed for job {job_id}: {error_msg}")
        logger.error(f"Traceback: {error_trace}")
        
        # Update job status to failed
        job_service.update_job_status(
            job_id, 
            JobStatus.FAILED, 
            error_message=error_msg
        )
        job_service.add_job_log(job_id, f"Training failed: {error_msg}")
        
        # Update task state
        self.update_state(
            state='FAILURE',
            meta={
                'error': error_msg,
                'traceback': error_trace,
                'job_id': job_id
            }
        )
        
        raise


@celery_app.task(bind=True, name='biomlstudio.ml_tasks.start_preprocessing_task')
def start_preprocessing_task(
    self, 
    job_id: int, 
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Start data preprocessing task.
    
    Args:
        job_id: Job ID
        config: Preprocessing configuration
        
    Returns:
        Dict: Task result
    """
    task_id = self.request.id
    logger.info(f"Starting preprocessing task {task_id} for job {job_id}")
    
    job_service = JobService()
    
    try:
        # Update job status
        job_service.update_job_status(job_id, JobStatus.RUNNING)
        job_service.add_job_log(job_id, f"Preprocessing task started: {task_id}")
        
        # Preprocessing steps would go here
        # This is a simplified implementation
        
        steps = [
            "Data validation",
            "Feature extraction", 
            "Data cleaning",
            "Feature scaling",
            "Data splitting"
        ]
        
        for i, step in enumerate(steps):
            progress = ((i + 1) / len(steps)) * 100
            job_service.update_job_progress(job_id, progress, step)
            
            self.update_state(
                state='PROGRESS',
                meta={
                    'current': int(progress),
                    'total': 100,
                    'status': f'Processing: {step}'
                }
            )
            
            # Simulate processing time
            import time
            time.sleep(2)
        
        # Complete job
        result_metrics = {
            'preprocessing_steps': len(steps),
            'processed_features': config.get('feature_count', 0),
            'output_path': config.get('output_path')
        }
        
        job_service.update_job_status(
            job_id,
            JobStatus.COMPLETED,
            metrics=result_metrics
        )
        job_service.add_job_log(job_id, "Preprocessing completed successfully")
        
        logger.info(f"Preprocessing task completed for job {job_id}")
        
        return {
            'status': 'completed',
            'job_id': job_id,
            'metrics': result_metrics,
            'task_id': task_id
        }
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        
        logger.error(f"Preprocessing task failed for job {job_id}: {error_msg}")
        
        job_service.update_job_status(
            job_id,
            JobStatus.FAILED,
            error_message=error_msg
        )
        job_service.add_job_log(job_id, f"Preprocessing failed: {error_msg}")
        
        self.update_state(
            state='FAILURE',
            meta={
                'error': error_msg,
                'traceback': error_trace,
                'job_id': job_id
            }
        )
        
        raise


@celery_app.task(bind=True, name='biomlstudio.ml_tasks.evaluate_model_task')
def evaluate_model_task(
    self,
    model_id: int,
    test_dataset_path: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate trained model on test dataset.
    
    Args:
        model_id: Model ID to evaluate
        test_dataset_path: Path to test dataset
        config: Evaluation configuration
        
    Returns:
        Dict: Evaluation results
    """
    task_id = self.request.id
    logger.info(f"Starting model evaluation task {task_id} for model {model_id}")
    
    ml_service = MLService()

    try:
        self.update_state(
            state='PROGRESS',
            meta={'current': 0, 'total': 100, 'status': 'Evaluating model...'}
        )

        eval_out = ml_service.evaluate_model(
            model_id=model_id,
            test_dataset_path=test_dataset_path,
            config=config,
        )

        self.update_state(
            state='PROGRESS',
            meta={'current': 100, 'total': 100, 'status': 'Evaluation completed'}
        )

        logger.info(f"Model evaluation completed for model {model_id}")

        return {
            'status': 'completed',
            'model_id': model_id,
            'results': eval_out,
            'task_id': task_id
        }
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Model evaluation failed for model {model_id}: {error_msg}")
        
        self.update_state(
            state='FAILURE',
            meta={'error': error_msg, 'model_id': model_id}
        )
        
        raise


@celery_app.task(bind=True, name='biomlstudio.ml_tasks.predict_batch_task')
def predict_batch_task(
    self,
    model_id: int,
    input_data_path: str,
    output_path: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Perform batch predictions with trained model.
    
    Args:
        model_id: Model ID for predictions
        input_data_path: Path to input data
        output_path: Path to save predictions
        config: Prediction configuration
        
    Returns:
        Dict: Prediction results
    """
    task_id = self.request.id
    logger.info(f"Starting batch prediction task {task_id} for model {model_id}")
    
    ml_service = MLService()
    
    try:
        self.update_state(
            state='PROGRESS',
            meta={'current': 0, 'total': 100, 'status': 'Loading data...'}
        )
        
        # Batch prediction logic would be implemented here
        # This is a simplified placeholder
        
        prediction_results = {
            'model_id': model_id,
            'predictions_count': 1000,
            'output_path': output_path,
            'batch_size': config.get('batch_size', 100)
        }
        
        self.update_state(
            state='PROGRESS',
            meta={'current': 100, 'total': 100, 'status': 'Predictions completed'}
        )
        
        logger.info(f"Batch prediction completed for model {model_id}")
        
        return {
            'status': 'completed',
            'model_id': model_id,
            'results': prediction_results,
            'task_id': task_id
        }
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Batch prediction failed for model {model_id}: {error_msg}")
        
        self.update_state(
            state='FAILURE',
            meta={'error': error_msg, 'model_id': model_id}
        )
        
        raise
