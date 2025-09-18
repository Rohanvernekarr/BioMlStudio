"""
Dataset service for managing biological datasets
"""
#Datasets: Upload, analyze, preview, validate

import hashlib
import logging
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional

from Bio import SeqIO
from Bio.SeqUtils import gc_fraction

from app.core.config import settings
from app.core.database import get_db_context
from app.models.dataset import Dataset
from app.utils.bioinformatics import (
    detect_sequence_type, validate_fasta_format, 
    calculate_sequence_composition
)

logger = logging.getLogger(__name__)


class DatasetService:
    """Service for dataset management and analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def analyze_dataset(
        self, 
        file_path: Path, 
        dataset_type: str
    ) -> Dict[str, Any]:
        """
        Analyze dataset and generate statistics.
        
        Args:
            file_path: Path to dataset file
            dataset_type: Type of dataset (dna, protein, rna, general)
            
        Returns:
            Dict: Dataset statistics
        """
        try:
            stats = {}
            file_extension = file_path.suffix.lower()
            
            # First check file extension to determine analysis method
            is_biological_extension = file_extension in ['.fasta', '.fa', '.fas', '.fastq', '.fq']
            is_general_extension = file_extension in ['.csv', '.tsv']
            
            if is_biological_extension:
                # If file is FASTA/FASTQ, analyze as biological data regardless of dataset_type
                stats = await self._analyze_biological_dataset(file_path, dataset_type)
            elif is_general_extension and dataset_type == 'general':
                # Only analyze as general dataset if explicitly marked as 'general' type
                stats = await self._analyze_general_dataset(file_path)
            elif dataset_type in ['dna', 'rna', 'protein']:
                # Fall back to biological analysis if dataset_type is specified
                stats = await self._analyze_biological_dataset(file_path, dataset_type)
            else:
                # Default to general analysis
                stats = await self._analyze_general_dataset(file_path)
            
            # Add file hash for integrity checking
            stats['file_hash'] = self._calculate_file_hash(file_path)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error analyzing dataset {file_path}: {e}")
            return {"error": str(e)}
    
    async def _analyze_biological_dataset(
        self, 
        file_path: Path, 
        dataset_type: str
    ) -> Dict[str, Any]:
        """Analyze biological sequence dataset"""
        file_extension = file_path.suffix.lower()
        
        if file_extension in ['.fasta', '.fa', '.fas']:
            return await self._analyze_fasta_file(file_path, dataset_type)
        elif file_extension in ['.fastq', '.fq']:
            return await self._analyze_fastq_file(file_path, dataset_type)
        elif file_extension in ['.csv', '.tsv']:
            return await self._analyze_sequence_csv(file_path, dataset_type)
        else:
            raise ValueError(f"Unsupported biological file format: {file_extension}")
    
    async def _analyze_fasta_file(
        self, 
        file_path: Path, 
        dataset_type: str
    ) -> Dict[str, Any]:
        """Analyze FASTA format file"""
        sequences = []
        sequence_lengths = []
        
        try:
            with open(file_path, 'r') as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    sequences.append(str(record.seq))
                    sequence_lengths.append(len(record.seq))
            
            if not sequences:
                raise ValueError("No valid sequences found in FASTA file")
            
            # Calculate sequence statistics
            stats = {
                "format": "fasta",
                "sequence_count": len(sequences),
                "total_sequences": len(sequences),
                "min_sequence_length": min(sequence_lengths),
                "max_sequence_length": max(sequence_lengths),
                "avg_sequence_length": sum(sequence_lengths) / len(sequence_lengths),
                "total_bases": sum(sequence_lengths)
            }
            
            # Add sequence-type specific stats
            if dataset_type in ['dna', 'rna']:
                gc_contents = [gc_fraction(seq) for seq in sequences[:100]]  # Sample first 100
                stats.update({
                    "avg_gc_content": sum(gc_contents) / len(gc_contents),
                    "min_gc_content": min(gc_contents),
                    "max_gc_content": max(gc_contents)
                })
                
                # Count nucleotides in sample
                sample_seq = ''.join(sequences[:10])  # First 10 sequences
                nucleotide_counts = {
                    'A': sample_seq.upper().count('A'),
                    'T': sample_seq.upper().count('T'),
                    'G': sample_seq.upper().count('G'),
                    'C': sample_seq.upper().count('C'),
                    'N': sample_seq.upper().count('N')
                }
                stats["nucleotide_composition"] = nucleotide_counts
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error analyzing FASTA file: {e}")
            raise
    
    async def _analyze_general_dataset(self, file_path: Path) -> Dict[str, Any]:
        """Analyze general CSV/TSV dataset"""
        try:
            # Detect delimiter
            with open(file_path, 'r') as f:
                first_line = f.readline()
                delimiter = '\t' if '\t' in first_line else ','
            
            # Read dataset
            df = pd.read_csv(file_path, delimiter=delimiter, nrows=10000)  # Sample first 10k rows
            
            stats = {
                "format": "csv" if delimiter == ',' else "tsv",
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "columns": list(df.columns),
                "column_types": df.dtypes.astype(str).to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
            }
            
            # Add sample data for preview
            stats["sample_data"] = df.head(5).to_dict('records')
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error analyzing general dataset: {e}")
            raise
    
    async def preview_dataset(
        self, 
        file_path: str, 
        dataset_type: str, 
        rows: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Generate preview data for dataset.
        
        Args:
            file_path: Path to dataset file
            dataset_type: Type of dataset
            rows: Number of rows to preview
            
        Returns:
            List: Preview data records
        """
        file_path = Path(file_path)
        
        try:
            if dataset_type in ['dna', 'rna', 'protein']:
                return await self._preview_biological_dataset(file_path, rows)
            else:
                return await self._preview_general_dataset(file_path, rows)
                
        except Exception as e:
            self.logger.error(f"Error previewing dataset: {e}")
            return []
    
    async def _preview_biological_dataset(
        self, 
        file_path: Path, 
        rows: int
    ) -> List[Dict[str, Any]]:
        """Preview biological sequence dataset"""
        preview_data = []
        
        if file_path.suffix.lower() in ['.fasta', '.fa', '.fas']:
            with open(file_path, 'r') as handle:
                for i, record in enumerate(SeqIO.parse(handle, "fasta")):
                    if i >= rows:
                        break
                    
                    preview_data.append({
                        "id": record.id,
                        "description": record.description,
                        "sequence": str(record.seq)[:100] + "..." if len(record.seq) > 100 else str(record.seq),
                        "length": len(record.seq)
                    })
        
        return preview_data
    
    async def _preview_general_dataset(
        self, 
        file_path: Path, 
        rows: int
    ) -> List[Dict[str, Any]]:
        """Preview general CSV/TSV dataset"""
        try:
            # Detect delimiter
            with open(file_path, 'r') as f:
                first_line = f.readline()
                delimiter = '\t' if '\t' in first_line else ','
            
            df = pd.read_csv(file_path, delimiter=delimiter, nrows=rows)
            return df.to_dict('records')
            
        except Exception as e:
            self.logger.error(f"Error previewing general dataset: {e}")
            return []
    
    async def validate_dataset(
        self, 
        file_path: str, 
        dataset_type: str
    ) -> Dict[str, Any]:
        """
        Validate dataset format and content.
        
        Args:
            file_path: Path to dataset file
            dataset_type: Expected dataset type
            
        Returns:
            Dict: Validation results
        """
        file_path = Path(file_path)
        
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "format_detected": None,
            "sequence_type": None
        }
        
        try:
            if dataset_type in ['dna', 'rna', 'protein']:
                return await self._validate_biological_dataset(file_path, dataset_type)
            else:
                return await self._validate_general_dataset(file_path)
                
        except Exception as e:
            validation_results["is_valid"] = False
            validation_results["errors"].append(str(e))
            return validation_results
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    async def get_dataset_by_id(self, dataset_id: int) -> Optional[Dataset]:
        """Get dataset by ID"""
        with get_db_context() as db:
            return db.query(Dataset).filter(Dataset.id == dataset_id).first()
    
    async def delete_dataset_files(self, dataset_id: int) -> bool:
        """Delete dataset files from storage"""
        dataset = await self.get_dataset_by_id(dataset_id)
        
        if not dataset:
            return False
        
        try:
            file_path = Path(dataset.file_path)
            if file_path.exists():
                file_path.unlink()
            
            self.logger.info(f"Dataset files deleted for dataset {dataset_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting dataset files: {e}")
            return False
