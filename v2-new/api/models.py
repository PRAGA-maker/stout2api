from pydantic import BaseModel, validator, Field
from typing import List, Optional
from enum import Enum
import re

class TranslationType(str, Enum):
    FORWARD = "forward"  # SMILES -> IUPAC
    REVERSE = "reverse"  # IUPAC -> SMILES

class InferenceRequest(BaseModel):
    input_text: str = Field(..., min_length=1, max_length=1000, description="Input SMILES or IUPAC name")
    translation_type: TranslationType

    @validator('input_text')
    def validate_input_text(cls, v):
        """Validate input text based on translation type"""
        if not v or not v.strip():
            raise ValueError("Input text cannot be empty or whitespace only")
        
        # Remove leading/trailing whitespace
        v = v.strip()
        
        # Basic length validation
        if len(v) > 1000:
            raise ValueError("Input text too long (max 1000 characters)")
        
        return v

class InferenceResponse(BaseModel):
    input: str
    output: str
    translation_type: str
    success: bool
    processing_time: Optional[float] = None
    error: Optional[str] = None
    warnings: Optional[List[str]] = None

class BatchInferenceRequest(BaseModel):
    inputs: List[str] = Field(..., min_items=1, max_items=1000, description="List of input SMILES or IUPAC names")
    translation_type: TranslationType

    @validator('inputs')
    def validate_inputs(cls, v):
        """Validate batch inputs"""
        if not v:
            raise ValueError("Input list cannot be empty")
        
        if len(v) > 1000:
            raise ValueError("Too many inputs (max 1000)")
        
        # Validate each input
        validated_inputs = []
        for i, input_text in enumerate(v):
            if not input_text or not input_text.strip():
                raise ValueError(f"Input at index {i} cannot be empty or whitespace only")
            
            if len(input_text.strip()) > 1000:
                raise ValueError(f"Input at index {i} too long (max 1000 characters)")
            
            validated_inputs.append(input_text.strip())
        
        return validated_inputs

class BatchInferenceResponse(BaseModel):
    results: List[InferenceResponse]
    total_processed: int
    successful: int
    failed: int
    warnings: Optional[List[str]] = None

class EvaluationRequest(BaseModel):
    inputs: List[str] = Field(..., min_items=1, max_items=1000, description="List of input SMILES or IUPAC names")
    translation_type: TranslationType
    include_roundtrip: bool = False
    return_visualization: bool = False
    return_csv: bool = False

    @validator('inputs')
    def validate_evaluation_inputs(cls, v):
        """Validate evaluation inputs"""
        if not v:
            raise ValueError("Input list cannot be empty")
        
        if len(v) > 1000:
            raise ValueError("Too many inputs for evaluation (max 1000)")
        
        # Validate each input
        validated_inputs = []
        for i, input_text in enumerate(v):
            if not input_text or not input_text.strip():
                raise ValueError(f"Input at index {i} cannot be empty or whitespace only")
            
            if len(input_text.strip()) > 1000:
                raise ValueError(f"Input at index {i} too long (max 1000 characters)")
            
            validated_inputs.append(input_text.strip())
        
        return validated_inputs

class EvaluationResponse(BaseModel):
    metrics: dict
    results: List[dict]
    visualization_path: Optional[str] = None
    csv_path: Optional[str] = None
    warnings: Optional[List[str]] = None 