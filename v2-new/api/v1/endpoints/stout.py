from fastapi import APIRouter, HTTPException, Query, status
from typing import List, Optional
import time
import csv
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
import logging
from api.models import (
    InferenceRequest, InferenceResponse, BatchInferenceRequest, 
    BatchInferenceResponse, EvaluationRequest, EvaluationResponse,
    TranslationType
)
from api.stout_service import stout_service
import ray
import gc

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stout", tags=["STOUT"])

INFERENCE_TIMEOUT = 3000  # seconds
MAX_INPUT_LENGTH = 1000
MAX_BATCH_SIZE = 1000
MAX_CONCURRENT_REQUESTS = 3  # Maximum concurrent requests for dynamic scaling

# Ray is initialized in main.py at startup

# Import dynamic scaling components
from api.utils.resource_manager import resource_manager
from api.utils.dynamic_actor_factory import actor_factory

# Global actor reference (will be dynamically created)
inference_actor = None


def calculate_tanimoto_similarity(smiles1: str, smiles2: str) -> float:
    """Calculate Tanimoto similarity between two SMILES strings."""
    try:
        if not smiles1 or not smiles2:
            return 0.0
            
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return 0.0
        
        fp1 = Chem.RDKFingerprint(mol1)
        fp2 = Chem.RDKFingerprint(mol2)
        
        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
        return similarity
    except Exception as e:
        logger.warning(f"Tanimoto similarity calculation failed: {e}")
        return 0.0

@router.get("/health")
def health_check():
    """Health check endpoint with dynamic scaling status"""
    try:
        ray_status = "initialized" if ray.is_initialized() else "not_initialized"
        
        # Get dynamic scaling status
        system_status = resource_manager.get_system_status()
        
        return {
            "status": "healthy", 
            "models_loaded": stout_service.models_loaded,
            "cdk_available": stout_service.cdk_available,
            "ray_status": ray_status,
            "dynamic_scaling": {
                "active_requests": system_status["active_requests"],
                "scaling_mode": system_status["current_config"]["scaling_mode"],
                "resource_utilization": {
                    "cpu_percent": system_status["system_resources"]["cpu_usage_percent"],
                    "memory_percent": system_status["system_resources"]["memory_usage_percent"]
                }
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}

@router.post("/reset-scaling")
def reset_scaling():
    """Emergency reset of dynamic scaling system"""
    try:
        # Reset resource manager
        old_count = resource_manager.reset_request_count()
        
        # Force cleanup all actors and resources
        actor_factory.force_resource_cleanup()
        
        return {
            "status": "reset_successful",
            "previous_active_requests": old_count,
            "message": "Dynamic scaling system has been reset and resources cleaned up"
        }
    except Exception as e:
        logger.error(f"Failed to reset scaling system: {e}")
        return {
            "status": "reset_failed",
            "error": str(e)
        }

@router.get("/ray-status")
def ray_status():
    """Check Ray cluster status and resource usage with dynamic scaling info"""
    try:
        if not ray.is_initialized():
            return {
                "ray_status": "not_initialized",
                "error": "Ray is not initialized"
            }
        
        # Get comprehensive system status including dynamic scaling
        system_status = resource_manager.get_system_status()
        actor_status = actor_factory.get_actor_status()
        
        # Get Ray cluster resources
        cluster_resources = ray.cluster_resources()
        available_resources = ray.available_resources()
        
        # Get Ray dashboard URL if available
        dashboard_url = None
        try:
            dashboard_url = f"http://localhost:8265"  # Default Ray dashboard port
        except Exception as e:
            logger.debug(f"Dashboard not available: {e}")
        
        # Get additional Ray metrics if available
        ray_metrics = {}
        try:
            nodes = ray.nodes()
            ray_metrics["node_count"] = len(nodes)
            ray_metrics["worker_count"] = len([node for node in nodes if node.get("Alive", False)])
        except Exception as e:
            logger.debug(f"Could not get additional Ray metrics: {e}")
            ray_metrics = {"error": "Could not retrieve additional metrics"}
        
        return {
            "ray_status": "initialized",
            "dynamic_scaling": {
                "system_resources": system_status["system_resources"],
                "active_requests": system_status["active_requests"],
                "current_config": system_status["current_config"],
                "actor_status": actor_status
            },
            "cluster_resources": {
                "cpu": cluster_resources.get("CPU", 0),
                "memory_gb": round(cluster_resources.get("memory", 0) / (1024**3), 2),
                "object_store_memory_gb": round(cluster_resources.get("object_store_memory", 0) / (1024**3), 2)
            },
            "available_resources": {
                "cpu": available_resources.get("CPU", 0),
                "memory_gb": round(available_resources.get("memory", 0) / (1024**3), 2)
            },
            "dashboard_url": dashboard_url,
            "dashboard_available": dashboard_url is not None,
            "metrics": ray_metrics
        }
    except Exception as e:
        logger.error(f"Ray status check failed: {e}")
        return {
            "ray_status": "error",
            "error": str(e)
        }

@router.post("/inference", response_model=InferenceResponse)
def single_inference(request: InferenceRequest):
    """Single molecule inference using dynamic scaling Ray actor"""
    try:
        start_time = time.time()
        warnings = []
        
        # Register request start and get optimal actor configuration
        config = resource_manager.register_request_start()
        logger.info(f"Starting inference with config: {config}")
        
        try:
            # Get optimal actor for current load
            actor = actor_factory.get_optimal_actor()
            
            # Call the actor's method
            future = actor.infer.remote(request.input_text, request.translation_type)
            try:
                output = ray.get(future, timeout=INFERENCE_TIMEOUT)
            except ray.exceptions.GetTimeoutError:
                raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=f"Inference exceeded {INFERENCE_TIMEOUT}s timeout.")
            
        finally:
            # Always register request end, even if inference fails
            resource_manager.register_request_end()
        
        processing_time = time.time() - start_time
        success = bool(output and output.strip() and output != "Invalid SMILES input provided." and output != "Could not canonicalize SMILES input." and output != "Translation produced empty result.")
        
        if output and len(output) > 200:
            warnings.append("Generated output is very long")
        if not success and not output:
            error_msg = "Translation failed or produced invalid output"
        elif not success:
            error_msg = output
        else:
            error_msg = None
            
        return InferenceResponse(
            input=request.input_text,
            output=output,
            translation_type=request.translation_type.value,
            success=success,
            processing_time=processing_time,
            error=error_msg,
            warnings=warnings if warnings else None
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Single inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/batch", response_model=BatchInferenceResponse)
def batch_inference(request: BatchInferenceRequest):
    """Batch inference for multiple molecules using dynamic scaling Ray actors"""
    if len(request.inputs) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=f"Batch size exceeds maximum allowed ({MAX_BATCH_SIZE})")
    
    results = []
    successful = 0
    failed = 0
    warnings = []
    logger.info(f"Starting batch inference for {len(request.inputs)} inputs")
    
    # Register batch request start
    config = resource_manager.register_request_start()
    logger.info(f"Starting batch inference with config: {config}")
    
    try:
        # Submit all tasks to dynamic Ray actors
        ray_futures = []
        for i, input_text in enumerate(request.inputs):
            if len(input_text) > MAX_INPUT_LENGTH:
                results.append(InferenceResponse(
                    input=input_text,
                    output="",
                    translation_type=request.translation_type.value,
                    success=False,
                    processing_time=0.0,
                    error=f"Input at index {i} exceeds max length ({MAX_INPUT_LENGTH})",
                    warnings=[]
                ))
                failed += 1
                continue
            
            # Get optimal actor for each request (will be cached efficiently)
            actor = actor_factory.get_optimal_actor()
            ray_futures.append((i, input_text, actor.infer.remote(input_text, request.translation_type)))
        
        # Gather results as they complete
        for i, input_text, future in ray_futures:
            try:
                output = ray.get(future, timeout=INFERENCE_TIMEOUT)
                gc.collect()
                processing_time = 0.0  # Optionally, you can track per-task time if needed
                success = bool(output and output.strip() and output != "Invalid SMILES input provided." and output != "Could not canonicalize SMILES input." and output != "Translation produced empty result.")
                item_warnings = []
                if output and len(output) > 200:
                    item_warnings.append("Generated output is very long")
                if not success and not output:
                    error_msg = "Translation failed or produced invalid output"
                elif not success:
                    error_msg = output
                else:
                    error_msg = None
                results.append(InferenceResponse(
                    input=input_text,
                    output=output,
                    translation_type=request.translation_type.value,
                    success=success,
                    processing_time=processing_time,
                    error=error_msg,
                    warnings=item_warnings if item_warnings else None
                ))
                if success:
                    successful += 1
                else:
                    failed += 1
            except ray.exceptions.GetTimeoutError:
                results.append(InferenceResponse(
                    input=input_text,
                    output="",
                    translation_type=request.translation_type.value,
                    success=False,
                    processing_time=INFERENCE_TIMEOUT,
                    error=f"Inference exceeded {INFERENCE_TIMEOUT}s timeout.",
                    warnings=None
                ))
                failed += 1
            except Exception as e:
                logger.error(f"Batch inference failed for input {i}: {e}")
                results.append(InferenceResponse(
                    input=input_text,
                    output="",
                    translation_type=request.translation_type.value,
                    success=False,
                    error=f"Processing error: {str(e)}",
                    warnings=None
                ))
                failed += 1
    finally:
        # Always register batch request end
        resource_manager.register_request_end()
    
    if failed > 0:
        warnings.append(f"{failed} out of {len(request.inputs)} inputs failed")
    if successful == 0:
        warnings.append("No successful translations in batch")
    logger.info(f"Batch inference completed: {successful} successful, {failed} failed")
    return BatchInferenceResponse(
        results=results,
        total_processed=len(request.inputs),
        successful=successful,
        failed=failed,
        warnings=warnings if warnings else None
    )

@router.post("/evaluate", response_model=EvaluationResponse)
def evaluate_model(
    request: EvaluationRequest,
    return_visualization: bool = Query(False, description="Return visualization path"),
    return_csv: bool = Query(False, description="Return CSV results path")
):
    """Evaluate model performance on batch data with comprehensive error handling"""
    
    logger.info(f"Starting evaluation for {len(request.inputs)} inputs")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"evaluation_results_{timestamp}"
    # Use absolute path to ensure files are created in the correct location
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(script_dir, output_dir)
    logger.info(f"Creating output directory: {output_dir}")
    
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create output directory: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create output directory: {e}")
    
    results = []
    metrics = {
        "tanimoto_similarities": [],
        "exact_matches": 0,
        "total_processed": 0,
        "successful": 0,
        "failed": 0
    }
    warnings = []
    
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
            
            # Determine success
            success = bool(predicted_output and predicted_output.strip() and 
                          predicted_output != "Invalid SMILES input provided." and
                          predicted_output != "Could not canonicalize SMILES input." and
                          predicted_output != "Translation produced empty result.")
            
            result = {
                "index": i,
                "input": input_text,
                "predicted_output": predicted_output,
                "roundtrip_output": roundtrip_output,
                "tanimoto_similarity": tanimoto,
                "exact_match": exact_match,
                "processing_time": processing_time,
                "success": success
            }
            
            results.append(result)
            metrics["total_processed"] += 1
            
            if success:
                metrics["successful"] += 1
            else:
                metrics["failed"] += 1
            
            if request.include_roundtrip:
                metrics["tanimoto_similarities"].append(tanimoto)
                if exact_match:
                    metrics["exact_matches"] += 1
                    
            gc.collect()
        except Exception as e:
            logger.error(f"Evaluation failed for input {i}: {e}")
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
            metrics["failed"] += 1
    
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
    
    # Add warnings
    if metrics["failed"] > 0:
        warnings.append(f"{metrics['failed']} out of {metrics['total_processed']} inputs failed")
    
    if metrics["successful"] == 0:
        warnings.append("No successful translations in evaluation")
    
    # Generate CSV if requested
    csv_path = None
    if return_csv:
        csv_path = os.path.join(output_dir, "evaluation_results.csv")
        logger.info(f"Writing CSV to: {csv_path}")
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['index', 'input', 'predicted_output', 'roundtrip_output', 
                             'tanimoto_similarity', 'exact_match', 
                             'processing_time', 'success', 'error']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            logger.info(f"Successfully wrote CSV to: {csv_path}")
        except Exception as e:
            logger.error(f"Error writing CSV: {e}")
            csv_path = None
            warnings.append(f"Failed to write CSV: {e}")
    
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
                logger.info(f"Successfully created visualization: {viz_path}")
            else:
                # Create a simple message if no valid results
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                ax.text(0.5, 0.5, 'No valid SMILES pairs found for visualization', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.axis('off')
                plt.savefig(viz_path, dpi=300, bbox_inches='tight')
                plt.close()
                warnings.append("No valid SMILES pairs found for visualization")
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            viz_path = None
            warnings.append(f"Failed to create visualization: {e}")
    
    logger.info(f"Evaluation completed: {metrics['successful']} successful, {metrics['failed']} failed")
    
    return EvaluationResponse(
        metrics=metrics,
        results=results,
        visualization_path=viz_path,
        csv_path=csv_path,
        warnings=warnings if warnings else None
    )

@router.get("/scaling-status")
def scaling_status():
    """Get detailed dynamic scaling status and resource utilization"""
    try:
        system_status = resource_manager.get_system_status()
        actor_status = actor_factory.get_actor_status()
        
        return {
            "dynamic_scaling": {
                "system_resources": system_status["system_resources"],
                "active_requests": system_status["active_requests"],
                "current_config": system_status["current_config"],
                "scaling_mode": system_status["current_config"]["scaling_mode"],
                "resource_utilization": {
                    "cpu_utilization": system_status["system_resources"]["cpu_usage_percent"],
                    "memory_utilization": system_status["system_resources"]["memory_usage_percent"],
                    "efficiency_score": _calculate_efficiency_score(system_status)
                }
            },
            "actor_management": {
                "cached_actors": actor_status["cached_actors"],
                "actor_configs": actor_status["actor_configs"]
            },
            "ray_cluster": system_status["ray_resources"]
        }
    except Exception as e:
        logger.error(f"Scaling status check failed: {e}")
        return {
            "error": "Failed to get scaling status",
            "detail": str(e)
        }

def _calculate_efficiency_score(system_status):
    """Calculate resource utilization efficiency score (0-100)"""
    try:
        cpu_util = system_status["system_resources"]["cpu_usage_percent"]
        mem_util = system_status["system_resources"]["memory_usage_percent"]
        active_reqs = system_status["active_requests"]
        
        # Base efficiency on resource utilization
        base_efficiency = (cpu_util + mem_util) / 2
        
        # Bonus for handling multiple requests efficiently
        if active_reqs > 0:
            request_efficiency = min(100, (active_reqs / 3) * 100)  # Max efficiency at 3 requests
            return min(100, (base_efficiency + request_efficiency) / 2)
        else:
            return base_efficiency
    except Exception:
        return 0.0

@router.get("/validate")
def validate_input(
    input_text: str = Query(..., description="Input text to validate"),
    translation_type: TranslationType = Query(..., description="Type of translation")
):
    """Validate input text with comprehensive validation"""
    try:
        validation_result = {
            "input": input_text,
            "translation_type": translation_type.value,
            "is_valid": False,
            "validation_type": "",
            "warnings": [],
            "error": None
        }
        
        if translation_type == TranslationType.FORWARD:
            # Validate SMILES
            is_valid = stout_service.is_valid_smiles(input_text)
            validation_result["is_valid"] = is_valid
            validation_result["validation_type"] = "SMILES validation"
            
            if not is_valid:
                validation_result["error"] = "SMILES failed RDKit validation"
            
            # Additional SMILES checks
            if len(input_text) > 500:
                validation_result["warnings"].append("SMILES is very long")
            
            # Check for potentially invalid characters
            import re
            smiles_pattern = r'^[A-Za-z0-9@+\-\[\]\(\)=#$%:\.]+$'
            if not re.match(smiles_pattern, input_text):
                validation_result["warnings"].append("SMILES contains potentially invalid characters")
                
        else:
            # For IUPAC, just check if it's not empty
            is_valid = len(input_text.strip()) > 0
            validation_result["is_valid"] = is_valid
            validation_result["validation_type"] = "IUPAC format check"
            
            if not is_valid:
                validation_result["error"] = "IUPAC name cannot be empty"
            
            # Additional IUPAC checks
            if len(input_text) > 500:
                validation_result["warnings"].append("IUPAC name is very long")
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return {
            "input": input_text,
            "translation_type": translation_type.value,
            "is_valid": False,
            "validation_type": "Error during validation",
            "error": str(e),
            "warnings": []
        } 