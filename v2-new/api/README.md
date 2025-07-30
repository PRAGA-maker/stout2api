# STOUT API

A FastAPI-based REST API for STOUT (SMILES to IUPAC and IUPAC to SMILES) translation.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables (optional):
```bash
export AZURE_CONNECTION_STRING="your_azure_connection_string"
export API_HOST="0.0.0.0"
export API_PORT="8000"
export DEBUG="True"
```

## Running the API

### Using the run script:
```bash
python run.py --host 0.0.0.0 --port 8000 --debug
```

### Using uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Command line arguments:
- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Port to bind to (default: 8000)
- `--reload`: Enable auto-reload for development
- `--workers`: Number of worker processes
- `--debug`: Enable debug mode

## API Endpoints

### Health Check
```
GET /stout/health
```

### Single Inference
```
POST /stout/inference
```
Body:
```json
{
    "input_text": "CCO",
    "translation_type": "forward"
}
```

### Batch Inference
```
POST /stout/batch
```
Body:
```json
{
    "inputs": ["CCO", "CCCC", "c1ccccc1"],
    "translation_type": "forward"
}
```

### Model Evaluation
```
POST /stout/evaluate?return_visualization=true&return_csv=true
```
Body:
```json
{
    "inputs": ["CCO", "CCCC", "c1ccccc1"],
    "translation_type": "forward",
    "include_roundtrip": true
}
```

### Input Validation
```
GET /stout/validate?input_text=CCO&translation_type=forward
```

### Scaling Status
```
GET /stout/scaling-status
```

### Ray Status
```
GET /stout/ray-status
```

### Reset Scaling
```
POST /stout/reset-scaling
```



## Translation Types

- `forward`: SMILES → IUPAC
- `reverse`: IUPAC → SMILES

## Evaluation Metrics

When `include_roundtrip=true`:
- **BLEU Score**: For IUPAC roundtrip (IUPAC → SMILES → IUPAC)
- **Tanimoto Similarity**: For SMILES roundtrip (SMILES → IUPAC → SMILES)
- **Exact Match Rate**: Percentage of exact matches in roundtrip

## Response Formats

### Inference Response
```json
{
    "input": "CCO",
    "output": "ethanol",
    "translation_type": "forward",
    "success": true,
    "processing_time": 0.123
}
```



### Evaluation Response
```json
{
    "metrics": {
        "avg_bleu": 0.85,
        "avg_tanimoto": 0.92,
        "exact_match_rate": 0.75,
        "success_rate": 0.95
    },
    "results": [...],
    "visualization_path": "evaluation_results_20231201_120000/evaluation_visualization.png",
    "csv_path": "evaluation_results_20231201_120000/evaluation_results.csv"
}
```
