"""
API endpoints for data transformation and feature engineering.
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile
import uuid

from app.core.security import get_current_user
from app.schemas.user import User
from app.services.transformation_service import transformation_service
from app.utils.file_handlers import save_upload_file

router = APIRouter()

@router.post("/extract-metadata", response_model=Dict[str, Any])
async def extract_metadata(
    file: UploadFile = File(...),
    dataset_type: str = Form(...),
    current_user: User = Depends(get_current_user)
):
    """
    Extract metadata from a dataset file.
    
    Args:
        file: Dataset file (FASTA, FASTQ, CSV, etc.)
        dataset_type: Type of dataset (dna, rna, protein, general)
        
    Returns:
        Extracted metadata
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = Path(temp_file.name)
        
        # Extract metadata
        metadata = await transformation_service.extract_metadata(temp_path, dataset_type)
        
        # Clean up
        temp_path.unlink()
        
        return {
            "status": "success",
            "metadata": metadata
        }
        
    except Exception as e:
        if 'temp_path' in locals():
            temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/encode-sequences", response_model=Dict[str, Any])
async def encode_sequences(
    sequences: List[str] = Form(...),
    encoding: str = Form("onehot"),
    dataset_type: str = Form("dna"),
    current_user: User = Depends(get_current_user)
):
    """
    Encode biological sequences.
    
    Args:
        sequences: List of sequences to encode
        encoding: Encoding method (onehot, integer, kmer)
        dataset_type: Type of sequences (dna, rna, protein)
        
    Returns:
        Encoded sequences
    """
    try:
        encoded = await transformation_service.encode_sequences(
            sequences=sequences,
            encoding=encoding,
            **{"dataset_type": dataset_type}
        )
        
        return {
            "status": "success",
            "encoding": encoding,
            "encoded_sequences": encoded.tolist()  # Convert numpy array to list
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/normalize", response_model=Dict[str, Any])
async def normalize_data(
    file: UploadFile = File(...),
    method: str = Form("minmax"),
    target_length: Optional[int] = Form(None),
    current_user: User = Depends(get_current_user)
):
    """
    Normalize sequence data.
    
    Args:
        file: Input file with sequences
        method: Normalization method (minmax, zscore, length)
        target_length: Target length for sequence normalization (required if method=length)
        
    Returns:
        Normalized data
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = Path(temp_file.name)
        
        # Read sequences
        sequences = []
        with open(temp_path, 'r') as f:
            for record in SeqIO.parse(f, "fasta"):
                sequences.append(str(record.seq))
        
        # Normalize
        normalized = await transformation_service.normalize_sequences(
            sequences=sequences,
            method=method,
            target_length=target_length
        )
        
        # Clean up
        temp_path.unlink()
        
        return {
            "status": "success",
            "method": method,
            "normalized_data": normalized
        }
        
    except Exception as e:
        if 'temp_path' in locals():
            temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e))
