"""
ML service for machine learning operations and model management
"""

import joblib
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

from app.core.config import settings
from app.core.database import get_db_context
from app.models.ml_model import MLModel, ModelType, ModelFramework
from app.models.job import Job

logger = logging.getLogger(__name__)


class MLService:
    """Service for machine learning operations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model_storage_path = Path(settings.MODEL_STORAGE_PATH)
        self.model_storage_path.mkdir(exist_ok=True)
    
    async def train_classification_model(
        self, 
        job_id: int,
        dataset_path: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Train a classification model.
        
        Args:
            job_id: Associated job ID
            dataset_path: Path to training dataset
            config: Training configuration
            
        Returns:
            Dict: Training results and metrics
        """
        try:
            # Load and prepare data
            X, y = await self._load_and_prepare_data(
                dataset_path, 
                config.get('target_column', 'target'),
                config.get('feature_columns')
            )
            
            # Split data
            test_size = config.get('test_size', 0.2)
            random_state = config.get('random_state', 42)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Scale features if requested
            scaler = None
            if config.get('scale_features', True):
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            
            # Initialize model
            algorithm = config.get('algorithm', 'random_forest')
            model = self._get_classification_model(algorithm, config.get('hyperparameters', {}))
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = self._calculate_classification_metrics(y_test, y_pred, y_proba)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            metrics['cv_accuracy_mean'] = cv_scores.mean()
            metrics['cv_accuracy_std'] = cv_scores.std()
            
            # Feature importance
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_names = config.get('feature_names', [f'feature_{i}' for i in range(X.shape[1])])
                feature_importance = dict(zip(feature_names, model.feature_importances_))
            
            # Save model
            model_path = await self._save_model(job_id, {
                'model': model,
                'scaler': scaler,
                'feature_names': config.get('feature_names'),
                'target_names': list(np.unique(y)),
                'config': config
            })
            
            return {
                'model_path': str(model_path),
                'metrics': metrics,
                'feature_importance': feature_importance,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'algorithm': algorithm
            }
            
        except Exception as e:
            self.logger.error(f"Error training classification model: {e}")
            raise
    
    async def train_regression_model(
        self, 
        job_id: int,
        dataset_path: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Train a regression model.
        
        Args:
            job_id: Associated job ID
            dataset_path: Path to training dataset
            config: Training configuration
            
        Returns:
            Dict: Training results and metrics
        """
        try:
            # Load and prepare data
            X, y = await self._load_and_prepare_data(
                dataset_path, 
                config.get('target_column', 'target'),
                config.get('feature_columns')
            )
            
            # Split data
            test_size = config.get('test_size', 0.2)
            random_state = config.get('random_state', 42)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Scale features if requested
            scaler = None
            if config.get('scale_features', True):
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            
            # Initialize model
            algorithm = config.get('algorithm', 'random_forest')
            model = self._get_regression_model(algorithm, config.get('hyperparameters', {}))
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = self._calculate_regression_metrics(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            metrics['cv_r2_mean'] = cv_scores.mean()
            metrics['cv_r2_std'] = cv_scores.std()
            
            # Feature importance
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_names = config.get('feature_names', [f'feature_{i}' for i in range(X.shape[1])])
                feature_importance = dict(zip(feature_names, model.feature_importances_))
            
            # Save model
            model_path = await self._save_model(job_id, {
                'model': model,
                'scaler': scaler,
                'feature_names': config.get('feature_names'),
                'config': config
            })
            
            return {
                'model_path': str(model_path),
                'metrics': metrics,
                'feature_importance': feature_importance,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'algorithm': algorithm
            }
            
        except Exception as e:
            self.logger.error(f"Error training regression model: {e}")
            raise
    
    async def predict_with_model(
        self,
        model_id: int,
        input_data: List[Dict[str, Any]],
        return_probabilities: bool = False
    ) -> Dict[str, Any]:
        """
        Make predictions with a trained model.
        
        Args:
            model_id: Model ID
            input_data: Input data for prediction
            return_probabilities: Whether to return prediction probabilities
            
        Returns:
            Dict: Prediction results
        """
        try:
            # Load model
            model_info = await self._load_model(model_id)
            
            if not model_info:
                raise ValueError(f"Model {model_id} not found")
            
            model = model_info['model']
            scaler = model_info.get('scaler')
            feature_names = model_info.get('feature_names')
            
            # Prepare input data
            df = pd.DataFrame(input_data)
            
            if feature_names:
                # Ensure all required features are present
                missing_features = set(feature_names) - set(df.columns)
                if missing_features:
                    raise ValueError(f"Missing features: {missing_features}")
                
                X = df[feature_names].values
            else:
                X = df.values
            
            # Scale features if scaler was used during training
            if scaler:
                X = scaler.transform(X)
            
            # Make predictions
            predictions = model.predict(X)
            
            results = {
                'predictions': predictions.tolist(),
                'model_id': model_id,
                'input_count': len(input_data)
            }
            
            # Add probabilities for classification models
            if return_probabilities and hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)
                results['probabilities'] = probabilities.tolist()
            
            # Update model usage count
            await self._update_model_usage(model_id)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error making predictions with model {model_id}: {e}")
            raise
    
    async def _load_and_prepare_data(
        self,
        dataset_path: str,
        target_column: str,
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load and prepare dataset for training"""
        
        # Load dataset
        if dataset_path.endswith('.csv'):
            df = pd.read_csv(dataset_path)
        elif dataset_path.endswith('.tsv'):
            df = pd.read_csv(dataset_path, sep='\t')
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")
        
        # Prepare features and target
        if feature_columns:
            X = df[feature_columns].values
        else:
            # Use all columns except target
            feature_cols = [col for col in df.columns if col != target_column]
            X = df[feature_cols].values
        
        y = df[target_column].values
        
        # Handle categorical target for classification
        if not np.issubdtype(y.dtype, np.number):
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        return X, y
    
    def _get_classification_model(self, algorithm: str, hyperparameters: Dict[str, Any]):
        """Get classification model instance"""
        
        if algorithm == 'random_forest':
            return RandomForestClassifier(
                n_estimators=hyperparameters.get('n_estimators', 100),
                max_depth=hyperparameters.get('max_depth'),
                min_samples_split=hyperparameters.get('min_samples_split', 2),
                min_samples_leaf=hyperparameters.get('min_samples_leaf', 1),
                random_state=hyperparameters.get('random_state', 42)
            )
        elif algorithm == 'logistic_regression':
            return LogisticRegression(
                C=hyperparameters.get('C', 1.0),
                max_iter=hyperparameters.get('max_iter', 1000),
                random_state=hyperparameters.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unsupported classification algorithm: {algorithm}")
    
    def _get_regression_model(self, algorithm: str, hyperparameters: Dict[str, Any]):
        """Get regression model instance"""
        
        if algorithm == 'random_forest':
            return RandomForestRegressor(
                n_estimators=hyperparameters.get('n_estimators', 100),
                max_depth=hyperparameters.get('max_depth'),
                min_samples_split=hyperparameters.get('min_samples_split', 2),
                min_samples_leaf=hyperparameters.get('min_samples_leaf', 1),
                random_state=hyperparameters.get('random_state', 42)
            )
        elif algorithm == 'linear_regression':
            return LinearRegression()
        else:
            raise ValueError(f"Unsupported regression algorithm: {algorithm}")
    
    def _calculate_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Calculate classification metrics"""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }
        
        # Add AUC for binary classification
        if len(np.unique(y_true)) == 2 and y_proba is not None:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba[:, 1])
            
            # ROC curve data
            fpr, tpr, thresholds = roc_curve(y_true, y_proba[:, 1])
            metrics['roc_curve'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist()
            }
        
        return metrics
    
    def _calculate_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate regression metrics"""
        
        return {
            'r2_score': r2_score(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': np.mean(np.abs(y_true - y_pred))
        }
    
    async def _save_model(self, job_id: int, model_info: Dict[str, Any]) -> Path:
        """Save trained model to disk"""
        
        model_dir = self.model_storage_path / f"job_{job_id}"
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / "model.joblib"
        
        # Save using joblib for sklearn models
        joblib.dump(model_info, model_path)
        
        self.logger.info(f"Model saved to {model_path}")
        return model_path
    
    async def _load_model(self, model_id: int) -> Optional[Dict[str, Any]]:
        """Load trained model from disk"""
        
        with get_db_context() as db:
            model_record = db.query(MLModel).filter(MLModel.id == model_id).first()
            
            if not model_record or not model_record.artifact_path:
                return None
            
            try:
                model_info = joblib.load(model_record.artifact_path)
                return model_info
            except Exception as e:
                self.logger.error(f"Error loading model {model_id}: {e}")
                return None
    
    async def _update_model_usage(self, model_id: int) -> None:
        """Update model usage statistics"""
        
        with get_db_context() as db:
            model = db.query(MLModel).filter(MLModel.id == model_id).first()
            
            if model:
                model.increment_prediction_count()
                db.commit()
