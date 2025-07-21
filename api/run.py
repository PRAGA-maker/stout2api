#!/usr/bin/env python3
"""
STOUT API Server Runner
"""
import argparse
import uvicorn
from config import settings

def main():
    parser = argparse.ArgumentParser(description="STOUT API Server")
    parser.add_argument("--host", default=settings.API_HOST, help="Host to bind to")
    parser.add_argument("--port", type=int, default=settings.API_PORT, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    print(f"Starting STOUT API server on {args.host}:{args.port}")
    print(f"Debug mode: {args.debug}")
    print(f"Auto-reload: {args.reload}")
    print(f"Workers: {args.workers}")
    
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level="debug" if args.debug else "info"
    )

if __name__ == "__main__":
    main() 