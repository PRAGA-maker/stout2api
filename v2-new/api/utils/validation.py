"""
Validation utilities for STOUT API
"""
import re
import logging
from typing import Tuple, List, Optional
from rdkit import Chem

logger = logging.getLogger(__name__)

def validate_smiles_format(smiles: str) -> Tuple[bool, List[str]]:
    """
    Validate SMILES format and return validation status and warnings
    
    Args:
        smiles: SMILES string to validate
        
    Returns:
        Tuple of (is_valid, warnings)
    """
    warnings = []
    
    if not smiles or not smiles.strip():
        return False, ["Empty SMILES input"]
    
    smiles = smiles.strip()
    
    # Basic SMILES pattern validation
    # Allow common SMILES characters: atoms, bonds, parentheses, brackets, etc.
    smiles_pattern = r'^[A-Za-z0-9@+\-\[\]\(\)=#$%:\.]+$'
    if not re.match(smiles_pattern, smiles):
        warnings.append("SMILES contains potentially invalid characters")
    
    # Length validation
    if len(smiles) > 500:
        warnings.append("SMILES is very long, may cause performance issues")
    elif len(smiles) < 2:
        warnings.append("SMILES is very short, may be invalid")
    
    # Check for balanced parentheses and brackets
    if not _check_balanced_brackets(smiles):
        warnings.append("SMILES has unbalanced parentheses or brackets")
    
    # Check for common SMILES issues
    if _has_common_smiles_issues(smiles):
        warnings.append("SMILES may have formatting issues")
    
    # Try RDKit validation
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, ["SMILES failed RDKit validation"] + warnings
    except Exception as e:
        warnings.append(f"RDKit validation warning: {str(e)}")
    
    return True, warnings

def validate_iupac_format(iupac: str) -> Tuple[bool, List[str]]:
    """
    Validate IUPAC name format and return validation status and warnings
    
    Args:
        iupac: IUPAC name to validate
        
    Returns:
        Tuple of (is_valid, warnings)
    """
    warnings = []
    
    if not iupac or not iupac.strip():
        return False, ["Empty IUPAC input"]
    
    iupac = iupac.strip()
    
    # Length validation
    if len(iupac) > 500:
        warnings.append("IUPAC name is very long, may cause performance issues")
    elif len(iupac) < 3:
        warnings.append("IUPAC name is very short, may be invalid")
    
    # Check for basic IUPAC patterns
    # IUPAC names typically contain letters, numbers, hyphens, and spaces
    iupac_pattern = r'^[A-Za-z0-9\s\-\(\)\[\],]+$'
    if not re.match(iupac_pattern, iupac):
        warnings.append("IUPAC name contains potentially invalid characters")
    
    # Check for balanced parentheses
    if not _check_balanced_brackets(iupac):
        warnings.append("IUPAC name has unbalanced parentheses")
    
    # Check for common IUPAC issues
    if _has_common_iupac_issues(iupac):
        warnings.append("IUPAC name may have formatting issues")
    
    return True, warnings

def sanitize_smiles(smiles: str) -> str:
    """
    Sanitize SMILES string by removing problematic characters and normalizing
    
    Args:
        smiles: Raw SMILES string
        
    Returns:
        Sanitized SMILES string
    """
    if not smiles:
        return ""
    
    # Remove leading/trailing whitespace
    smiles = smiles.strip()
    
    # Replace common problematic characters
    replacements = {
        '\\/': '/',  # Fix escaped forward slashes
        '\\\\': '\\',  # Fix double backslashes
        '\t': '',  # Remove tabs
        '\n': '',  # Remove newlines
        '\r': '',  # Remove carriage returns
    }
    
    for old, new in replacements.items():
        smiles = smiles.replace(old, new)
    
    # Remove multiple spaces
    smiles = re.sub(r'\s+', '', smiles)
    
    return smiles

def sanitize_iupac(iupac: str) -> str:
    """
    Sanitize IUPAC name by normalizing whitespace and removing problematic characters
    
    Args:
        iupac: Raw IUPAC name
        
    Returns:
        Sanitized IUPAC name
    """
    if not iupac:
        return ""
    
    # Remove leading/trailing whitespace
    iupac = iupac.strip()
    
    # Replace problematic characters
    replacements = {
        '\t': ' ',  # Replace tabs with spaces
        '\n': ' ',  # Replace newlines with spaces
        '\r': ' ',  # Replace carriage returns with spaces
    }
    
    for old, new in replacements.items():
        iupac = iupac.replace(old, new)
    
    # Normalize multiple spaces to single spaces
    iupac = re.sub(r'\s+', ' ', iupac)
    
    return iupac

def _check_balanced_brackets(text: str) -> bool:
    """Check if parentheses and brackets are balanced"""
    stack = []
    brackets = {')': '(', ']': '[', '}': '{'}
    
    for char in text:
        if char in '([{':
            stack.append(char)
        elif char in ')]}':
            if not stack or stack.pop() != brackets[char]:
                return False
    
    return len(stack) == 0

def _has_common_smiles_issues(smiles: str) -> bool:
    """Check for common SMILES formatting issues"""
    issues = [
        r'[A-Z]{5,}',  # Five or more consecutive uppercase letters (likely invalid)
        r'[0-9]{3,}',  # Three or more consecutive digits (likely invalid)
        r'[=#]{2,}',   # Multiple consecutive bond symbols
        r'[\[\]]{2,}', # Multiple consecutive brackets
        r'[\(\)]{2,}', # Multiple consecutive parentheses
    ]
    
    for pattern in issues:
        if re.search(pattern, smiles):
            return True
    
    return False

def _has_common_iupac_issues(iupac: str) -> bool:
    """Check for common IUPAC formatting issues"""
    issues = [
        r'[A-Z]{10,}',  # Very long consecutive uppercase letters
        r'[0-9]{4,}',   # Very long consecutive digits
        r'[,\s]{3,}',   # Multiple consecutive commas or spaces
    ]
    
    for pattern in issues:
        if re.search(pattern, iupac):
            return True
    
    return False

def validate_batch_inputs(inputs: List[str], max_batch_size: int = 1000) -> Tuple[bool, List[str]]:
    """
    Validate a batch of inputs
    
    Args:
        inputs: List of input strings
        max_batch_size: Maximum allowed batch size
        
    Returns:
        Tuple of (is_valid, warnings)
    """
    warnings = []
    
    if not inputs:
        return False, ["Empty input list"]
    
    if len(inputs) > max_batch_size:
        return False, [f"Batch size {len(inputs)} exceeds maximum of {max_batch_size}"]
    
    # Check for duplicate inputs
    unique_inputs = set()
    duplicates = []
    for i, input_text in enumerate(inputs):
        if input_text in unique_inputs:
            duplicates.append(i)
        else:
            unique_inputs.add(input_text)
    
    if duplicates:
        warnings.append(f"Found {len(duplicates)} duplicate inputs at indices: {duplicates}")
    
    # Check for empty inputs
    empty_indices = [i for i, input_text in enumerate(inputs) if not input_text or not input_text.strip()]
    if empty_indices:
        warnings.append(f"Found {len(empty_indices)} empty inputs at indices: {empty_indices}")
    
    return True, warnings 