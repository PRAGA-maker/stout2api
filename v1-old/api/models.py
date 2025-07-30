from pydantic import BaseModel
from typing import List, Optional
from enum import Enum

class TranslationType(str, Enum):
    FORWARD = "forward"  # SMILES -> IUPAC
    REVERSE = "reverse"  # IUPAC -> SMILES

class InferenceRequest(BaseModel):
    input_text: str
    translation_type: TranslationType

class InferenceResponse(BaseModel):
    input: str
    output: str
    translation_type: str
    success: bool
    processing_time: Optional[float] = None
    error: Optional[str] = None

class BatchInferenceRequest(BaseModel):
    inputs: List[str]
    translation_type: TranslationType

class BatchInferenceResponse(BaseModel):
    results: List[InferenceResponse]
    total_processed: int
    successful: int

class EvaluationRequest(BaseModel):
    inputs: List[str]
    translation_type: TranslationType
    include_roundtrip: bool = False
    return_visualization: bool = False
    return_csv: bool = False

class EvaluationResponse(BaseModel):
    metrics: dict
    results: List[dict]
    visualization_path: Optional[str] = None
    csv_path: Optional[str] = None 