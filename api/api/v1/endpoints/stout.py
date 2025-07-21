from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import time
import csv
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from models import (
    InferenceRequest, InferenceResponse, BatchInferenceRequest, 
    BatchInferenceResponse, EvaluationRequest, EvaluationResponse,
    TranslationType
)
from stout_service import stout_service

router = APIRouter(prefix="/stout", tags=["STOUT"])

def calculate_tanimoto_similarity(smiles1: str, smiles2: str) -> float:
    """Calculate Tanimoto similarity between two SMILES strings."""
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return 0.0
        
        fp1 = Chem.RDKFingerprint(mol1)
        fp2 = Chem.RDKFingerprint(mol2)
        
        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
        return similarity
    except Exception:
        return 0.0



@router.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": stout_service.models_loaded}

@router.post("/inference", response_model=InferenceResponse)
def single_inference(request: InferenceRequest):
    """Single molecule inference"""
    try:
        start_time = time.time()
        
        # Handle empty input
        if not request.input_text.strip():
            return InferenceResponse(
                input=request.input_text,
                output="",
                translation_type=request.translation_type.value,
                success=True,
                processing_time=0.0
            )
        
        if request.translation_type == TranslationType.FORWARD:
            output = stout_service.translate_forward(request.input_text)
        else:
            output = stout_service.translate_reverse(request.input_text)
        
        processing_time = time.time() - start_time
        
        # Check if output is valid
        success = bool(output and output.strip())
        
        return InferenceResponse(
            input=request.input_text,
            output=output,
            translation_type=request.translation_type.value,
            success=success,
            processing_time=processing_time,
            error=None if success else "Translation failed or produced invalid output"
        )
    except Exception as e:
        return InferenceResponse(
            input=request.input_text,
            output="",
            translation_type=request.translation_type.value,
            success=False,
            error=str(e)
        )

@router.post("/batch", response_model=BatchInferenceResponse)
def batch_inference(request: BatchInferenceRequest):
    """Batch inference for multiple molecules"""
    results = []
    successful = 0
    
    for input_text in request.inputs:
        try:
            start_time = time.time()
            
            # Handle empty input
            if not input_text.strip():
                results.append(InferenceResponse(
                    input=input_text,
                    output="",
                    translation_type=request.translation_type.value,
                    success=True,
                    processing_time=0.0
                ))
                successful += 1
                continue
            
            if request.translation_type == TranslationType.FORWARD:
                output = stout_service.translate_forward(input_text)
            else:
                output = stout_service.translate_reverse(input_text)
            
            processing_time = time.time() - start_time
            
            # Check if output is valid
            success = bool(output and output.strip())
            
            results.append(InferenceResponse(
                input=input_text,
                output=output,
                translation_type=request.translation_type.value,
                success=success,
                processing_time=processing_time,
                error=None if success else "Translation failed or produced invalid output"
            ))
            
            if success:
                successful += 1
            
        except Exception as e:
            results.append(InferenceResponse(
                input=input_text,
                output="",
                translation_type=request.translation_type.value,
                success=False,
                error=str(e)
            ))
    
    return BatchInferenceResponse(
        results=results,
        total_processed=len(request.inputs),
        successful=successful
    )

@router.post("/evaluate", response_model=EvaluationResponse)
def evaluate_model(
    request: EvaluationRequest,
    return_visualization: bool = Query(False, description="Return visualization path"),
    return_csv: bool = Query(False, description="Return CSV results path")
):
    """Evaluate model performance on batch data"""
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"evaluation_results_{timestamp}"
    # Use absolute path to ensure files are created in the correct location
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(script_dir, output_dir)
    print(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    metrics = {
        "tanimoto_similarities": [],
        "exact_matches": 0,
        "total_processed": 0,
        "successful": 0
    }
    
    for i, input_text in enumerate(request.inputs):
        try:
            start_time = time.time()
            
            # Forward translation
            if request.translation_type == TranslationType.FORWARD:
                predicted_output = stout_service.translate_forward(input_text)
                original_input = input_text
                
                # Roundtrip if requested
                if request.include_roundtrip:
                    roundtrip_output = stout_service.iupac_to_smiles_opsin(predicted_output)
                    tanimoto = calculate_tanimoto_similarity(original_input, roundtrip_output)
                    exact_match = original_input == roundtrip_output
                else:
                    roundtrip_output = ""
                    tanimoto = 0.0
                    exact_match = False
                    
            else:  # REVERSE
                predicted_output = stout_service.translate_reverse(input_text)
                original_input = input_text
                
                # Roundtrip if requested
                if request.include_roundtrip:
                    roundtrip_output = stout_service.translate_forward(predicted_output)
                    tanimoto = 0.0  # Not applicable for IUPAC
                    exact_match = original_input == roundtrip_output
                else:
                    roundtrip_output = ""
                    tanimoto = 0.0
                    exact_match = False
            
            processing_time = time.time() - start_time
            
            result = {
                "index": i,
                "input": input_text,
                "predicted_output": predicted_output,
                "roundtrip_output": roundtrip_output,
                "tanimoto_similarity": tanimoto,
                "exact_match": exact_match,
                "processing_time": processing_time,
                "success": True
            }
            
            results.append(result)
            metrics["total_processed"] += 1
            metrics["successful"] += 1
            
            if request.include_roundtrip:
                metrics["tanimoto_similarities"].append(tanimoto)
                if exact_match:
                    metrics["exact_matches"] += 1
                    
        except Exception as e:
            result = {
                "index": i,
                "input": input_text,
                "predicted_output": "",
                "roundtrip_output": "",
                "tanimoto_similarity": 0.0,
                "exact_match": False,
                "processing_time": 0.0,
                "success": False,
                "error": str(e)
            }
            results.append(result)
            metrics["total_processed"] += 1
    
    # Calculate summary metrics
    if metrics["tanimoto_similarities"]:
        metrics["avg_tanimoto"] = sum(metrics["tanimoto_similarities"]) / len(metrics["tanimoto_similarities"])
    else:
        metrics["avg_tanimoto"] = 0.0
    
    if metrics["total_processed"] > 0:
        metrics["exact_match_rate"] = metrics["exact_matches"] / metrics["total_processed"]
        metrics["success_rate"] = metrics["successful"] / metrics["total_processed"]
    else:
        metrics["exact_match_rate"] = 0.0
        metrics["success_rate"] = 0.0
    
    # Generate CSV if requested
    csv_path = None
    if return_csv:
        csv_path = os.path.join(output_dir, "evaluation_results.csv")
        print(f"Attempting to write CSV to: {csv_path}")
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['index', 'input', 'predicted_output', 'roundtrip_output', 
                             'tanimoto_similarity', 'exact_match', 
                             'processing_time', 'success', 'error']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            print(f"Successfully wrote CSV to: {csv_path}")
        except Exception as e:
            print(f"Error writing CSV: {e}")
            csv_path = None
    
    # Generate visualization if requested
    viz_path = None
    if return_visualization and results and request.translation_type == TranslationType.FORWARD and request.include_roundtrip:
        viz_path = os.path.join(output_dir, "evaluation_visualization.png")
        try:
            # Filter successful results with valid SMILES for visualization
            valid_results = []
            for r in results:
                if (r["success"] and r["input"] and r["roundtrip_output"] and 
                    stout_service.is_valid_smiles(r["input"]) and 
                    stout_service.is_valid_smiles(r["roundtrip_output"])):
                    valid_results.append(r)
            
            if valid_results:
                # Limit to first 12 results for visualization
                display_results = valid_results[:12]
                n_results = len(display_results)
                
                # Calculate grid dimensions
                cols = 4
                rows = (n_results + cols - 1) // cols
                
                fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
                if rows == 1:
                    axes = axes.reshape(1, -1)
                
                for idx, result in enumerate(display_results):
                    row = idx // cols
                    col = idx % cols
                    ax = axes[row, col]
                    
                    try:
                        # Create RDKit molecules
                        from rdkit import Chem
                        from rdkit.Chem import Draw
                        
                        mol1 = Chem.MolFromSmiles(result["input"])
                        mol2 = Chem.MolFromSmiles(result["roundtrip_output"])
                        
                        if mol1 and mol2:
                            # Create side-by-side visualization
                            img = Draw.MolsToGridImage(
                                [mol1, mol2], 
                                molsPerRow=2,
                                subImgSize=(200, 200),
                                legends=[f'Original: {result["input"]}', f'Roundtrip: {result["roundtrip_output"]}'],
                                useSVG=False
                            )
                            
                            # Convert PIL image to matplotlib
                            ax.imshow(img)
                            ax.axis('off')
                            
                            # Add similarity score
                            similarity = result["tanimoto_similarity"]
                            ax.set_title(f'Similarity: {similarity:.3f}', fontsize=10)
                            
                    except Exception as e:
                        ax.text(0.5, 0.5, f'Error: {str(e)[:20]}...', 
                               ha='center', va='center', transform=ax.transAxes)
                        ax.axis('off')
                
                # Hide empty subplots
                for idx in range(n_results, rows * cols):
                    row = idx // cols
                    col = idx % cols
                    axes[row, col].axis('off')
                
                plt.suptitle(f'SMILES → IUPAC → SMILES Roundtrip Visualization\n'
                            f'Showing {n_results} successful translations', fontsize=14)
                plt.tight_layout()
                plt.savefig(viz_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                # Create a simple message if no valid results
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                ax.text(0.5, 0.5, 'No valid SMILES pairs found for visualization', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.axis('off')
                plt.savefig(viz_path, dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"Error creating visualization: {e}")
            viz_path = None
    
    return EvaluationResponse(
        metrics=metrics,
        results=results,
        visualization_path=viz_path,
        csv_path=csv_path
    )

@router.get("/validate")
def validate_input(
    input_text: str = Query(..., description="Input text to validate"),
    translation_type: TranslationType = Query(..., description="Type of translation")
):
    """Validate input text"""
    try:
        if translation_type == TranslationType.FORWARD:
            # Validate SMILES
            is_valid = stout_service.is_valid_smiles(input_text)
            return {
                "input": input_text,
                "translation_type": translation_type.value,
                "is_valid": is_valid,
                "validation_type": "SMILES validation"
            }
        else:
            # For IUPAC, just check if it's not empty
            is_valid = len(input_text.strip()) > 0
            return {
                "input": input_text,
                "translation_type": translation_type.value,
                "is_valid": is_valid,
                "validation_type": "IUPAC format check"
            }
    except Exception as e:
        return {
            "input": input_text,
            "translation_type": translation_type.value,
            "is_valid": False,
            "error": str(e)
        } 