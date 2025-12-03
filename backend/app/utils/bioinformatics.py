"""
Bioinformatics utility functions for sequence analysis
"""

import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction
from Bio.SeqUtils.ProtParam import ProteinAnalysis

logger = logging.getLogger(__name__)


def detect_sequence_type(sequence: str) -> str:
    """
    Detect the type of biological sequence.
    
    Args:
        sequence: Biological sequence string
        
    Returns:
        str: Sequence type ('dna', 'rna', 'protein', 'unknown')
    """
    sequence = sequence.upper().strip()
    
    if not sequence:
        return 'unknown'
    
    # Count nucleotides and amino acids
    nucleotides = set('ATCG')
    rna_nucleotides = set('AUCG')
    amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
    
    seq_chars = set(sequence)
    
    # Calculate composition ratios
    nucleotide_ratio = len(seq_chars & nucleotides) / len(seq_chars) if seq_chars else 0
    rna_ratio = len(seq_chars & rna_nucleotides) / len(seq_chars) if seq_chars else 0
    amino_ratio = len(seq_chars & amino_acids) / len(seq_chars) if seq_chars else 0
    
    # Determine sequence type based on composition
    if nucleotide_ratio > 0.9 and 'U' not in sequence:
        return 'dna'
    elif rna_ratio > 0.9 and 'U' in sequence:
        return 'rna'
    elif amino_ratio > 0.8:
        return 'protein'
    else:
        return 'unknown'


def validate_fasta_format(file_path: str) -> Dict[str, Any]:
    """
    Validate FASTA file format.
    
    Args:
        file_path: Path to FASTA file
        
    Returns:
        Dict: Validation results
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'sequence_count': 0,
        'status': 'valid'
    }
    
    try:
        with open(file_path, 'r') as handle:
            sequences = list(SeqIO.parse(handle, "fasta"))
            
            if not sequences:
                validation_result['is_valid'] = False
                validation_result['errors'].append("No valid sequences found in file")
                validation_result['status'] = 'invalid'
                return validation_result
            
            validation_result['sequence_count'] = len(sequences)
            
            # Check for common issues
            for i, record in enumerate(sequences):
                # Check for empty sequences
                if len(record.seq) == 0:
                    validation_result['warnings'].append(f"Empty sequence at record {i+1}")
                
                # Check for very short sequences
                if len(record.seq) < 10:
                    validation_result['warnings'].append(f"Very short sequence at record {i+1}")
                
                # Check for invalid characters
                seq_str = str(record.seq).upper()
                valid_chars = set('ATCGURYSWKMBDHVN-')  # Standard IUPAC codes
                invalid_chars = set(seq_str) - valid_chars
                
                if invalid_chars:
                    validation_result['warnings'].append(
                        f"Invalid characters {invalid_chars} in record {i+1}"
                    )
            
            # Set status based on warnings
            if validation_result['warnings']:
                validation_result['status'] = 'valid_with_warnings'
            
    except Exception as e:
        validation_result['is_valid'] = False
        validation_result['errors'].append(f"Error reading FASTA file: {str(e)}")
        validation_result['status'] = 'error'
    
    return validation_result


def calculate_sequence_composition(sequences: List[str], seq_type: str) -> Dict[str, Any]:
    """
    Calculate composition statistics for sequences.
    
    Args:
        sequences: List of sequence strings
        seq_type: Type of sequence ('dna', 'rna', 'protein')
        
    Returns:
        Dict: Composition statistics
    """
    if not sequences:
        return {}
    
    stats = {
        'sequence_count': len(sequences),
        'total_length': sum(len(seq) for seq in sequences),
        'avg_length': sum(len(seq) for seq in sequences) / len(sequences),
        'min_length': min(len(seq) for seq in sequences),
        'max_length': max(len(seq) for seq in sequences)
    }
    
    if seq_type in ['dna', 'rna']:
        # Nucleotide composition
        all_sequence = ''.join(sequences).upper()
        total_bases = len(all_sequence)
        
        base_counts = Counter(all_sequence)
        stats['composition'] = {
            base: count / total_bases * 100 
            for base, count in base_counts.items()
        }
        
        # GC content for sample of sequences
        gc_contents = []
        for seq in sequences[:1000]:  # Sample first 1000
            try:
                gc_content = gc_fraction(seq) * 100
                gc_contents.append(gc_content)
            except:
                continue
        
        if gc_contents:
            stats['gc_content'] = {
                'mean': np.mean(gc_contents),
                'std': np.std(gc_contents),
                'min': np.min(gc_contents),
                'max': np.max(gc_contents)
            }
    
    elif seq_type == 'protein':
        # Amino acid composition
        all_sequence = ''.join(sequences).upper()
        total_aa = len(all_sequence)
        
        aa_counts = Counter(all_sequence)
        stats['composition'] = {
            aa: count / total_aa * 100 
            for aa, count in aa_counts.items()
        }
        
        # Protein properties for sample
        molecular_weights = []
        isoelectric_points = []
        
        for seq in sequences[:100]:  # Sample first 100
            try:
                if re.match(r'^[ACDEFGHIKLMNPQRSTVWY]*$', seq.upper()):
                    analysis = ProteinAnalysis(seq)
                    molecular_weights.append(analysis.molecular_weight())
                    isoelectric_points.append(analysis.isoelectric_point())
            except:
                continue
        
        if molecular_weights:
            stats['molecular_weight'] = {
                'mean': np.mean(molecular_weights),
                'std': np.std(molecular_weights),
                'min': np.min(molecular_weights),
                'max': np.max(molecular_weights)
            }
        
        if isoelectric_points:
            stats['isoelectric_point'] = {
                'mean': np.mean(isoelectric_points),
                'std': np.std(isoelectric_points),
                'min': np.min(isoelectric_points),
                'max': np.max(isoelectric_points)
            }
    
    return stats


def generate_kmer_features(
    sequences: List[str], 
    k: int = 3,
    normalize: bool = True
) -> Dict[str, List[float]]:
    """
    Generate k-mer frequency features from sequences.
    
    Args:
        sequences: List of sequences
        k: K-mer size
        normalize: Whether to normalize frequencies
        
    Returns:
        Dict: K-mer features for each sequence
    """
    if not sequences:
        return {}
    
    # Get all possible k-mers from sequences
    all_kmers = set()
    kmer_counts = []
    
    for seq in sequences:
        seq = seq.upper()
        seq_kmers = defaultdict(int)
        
        # Extract k-mers from sequence
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            # Only include k-mers with valid nucleotides
            if re.match(r'^[ATCG]*$', kmer):
                seq_kmers[kmer] += 1
                all_kmers.add(kmer)
        
        kmer_counts.append(seq_kmers)
    
    # Convert to feature matrix
    all_kmers = sorted(list(all_kmers))
    features = {kmer: [] for kmer in all_kmers}
    
    for seq_kmers in kmer_counts:
        total_kmers = sum(seq_kmers.values()) if normalize else 1
        
        for kmer in all_kmers:
            count = seq_kmers.get(kmer, 0)
            frequency = count / total_kmers if total_kmers > 0 else 0
            features[kmer].append(frequency)
    
    return features


def convert_fasta_to_csv(
    fasta_path: str, 
    csv_path: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convert FASTA file to CSV format.
    
    Args:
        fasta_path: Input FASTA file path
        csv_path: Output CSV file path
        config: Conversion configuration
        
    Returns:
        Dict: Conversion results
    """
    try:
        sequences_data = []
        
        with open(fasta_path, 'r') as handle:
            for i, record in enumerate(SeqIO.parse(handle, "fasta")):
                seq_data = {
                    'sequence_id': record.id,
                    'sequence': str(record.seq),
                    'length': len(record.seq)
                }
                
                # Extract label from description if present
                description_parts = record.description.split()
                if len(description_parts) > 1:
                    seq_data['label'] = description_parts[-1]
                
                # Add sequence type detection
                seq_type = detect_sequence_type(str(record.seq))
                seq_data['sequence_type'] = seq_type
                
                # Add composition features if requested
                if config.get('add_composition', False):
                    if seq_type in ['dna', 'rna']:
                        seq_str = str(record.seq).upper()
                        total_len = len(seq_str)
                        
                        seq_data.update({
                            'gc_content': gc_fraction(seq_str) * 100,
                            'a_content': seq_str.count('A') / total_len * 100,
                            't_content': seq_str.count('T') / total_len * 100,
                            'c_content': seq_str.count('C') / total_len * 100,
                            'g_content': seq_str.count('G') / total_len * 100,
                            'n_content': seq_str.count('N') / total_len * 100
                        })
                
                # Add k-mer features if requested
                if config.get('add_kmers', False):
                    kmer_size = config.get('kmer_size', 3)
                    kmers = generate_kmer_features([str(record.seq)], kmer_size)
                    
                    for kmer, freqs in kmers.items():
                        seq_data[f'kmer_{kmer}'] = freqs[0] if freqs else 0
                
                sequences_data.append(seq_data)
                
                # Limit for large files
                if config.get('max_sequences') and i >= config['max_sequences']:
                    break
        
        # Create DataFrame and save
        df = pd.DataFrame(sequences_data)
        
        # Reorder columns: label first, then numeric features, drop metadata
        if 'label' in df.columns:
            other_cols = [col for col in df.columns if col not in ['sequence_id', 'sequence', 'sequence_type', 'label']]
            df = df[['label'] + other_cols]
        
        df.to_csv(csv_path, index=False)
        
        return {
            'success': True,
            'sequences_converted': len(sequences_data),
            'output_path': csv_path,
            'columns': list(df.columns)
        }
        
    except Exception as e:
        logger.error(f"Error converting FASTA to CSV: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def validate_biological_file(file_path: str, expected_type: str) -> Dict[str, Any]:
    """
    Validate biological file format and content.
    
    Args:
        file_path: Path to file
        expected_type: Expected sequence type
        
    Returns:
        Dict: Validation results
    """
    file_path = Path(file_path)
    
    validation = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'detected_format': None,
        'detected_type': None
    }
    
    try:
        # Detect file format
        if file_path.suffix.lower() in ['.fasta', '.fa', '.fas']:
            validation['detected_format'] = 'fasta'
            
            # Validate FASTA format
            fasta_validation = validate_fasta_format(str(file_path))
            validation['is_valid'] = fasta_validation['is_valid']
            validation['errors'].extend(fasta_validation['errors'])
            validation['warnings'].extend(fasta_validation['warnings'])
            
            # Check sequence type if file is valid
            if validation['is_valid'] and fasta_validation['sequence_count'] > 0:
                with open(file_path, 'r') as handle:
                    first_record = next(SeqIO.parse(handle, "fasta"))
                    validation['detected_type'] = detect_sequence_type(str(first_record.seq))
                    
                    # Check if detected type matches expected
                    if validation['detected_type'] != expected_type:
                        validation['warnings'].append(
                            f"Detected sequence type '{validation['detected_type']}' "
                            f"doesn't match expected type '{expected_type}'"
                        )
        
        elif file_path.suffix.lower() in ['.fastq', '.fq']:
            validation['detected_format'] = 'fastq'
            # FASTQ validation would be implemented here
            
        elif file_path.suffix.lower() in ['.csv', '.tsv']:
            validation['detected_format'] = 'tabular'
            # CSV/TSV validation would be implemented here
            
        else:
            validation['is_valid'] = False
            validation['errors'].append(f"Unsupported file format: {file_path.suffix}")
    
    except Exception as e:
        validation['is_valid'] = False
        validation['errors'].append(f"Error validating file: {str(e)}")
    
    return validation
