import os
from typing import Optional

class Settings:
    # Azure configuration
    AZURE_CONNECTION_STRING: Optional[str] = os.getenv('AZURE_CONNECTION_STRING')
    AZURE_STORAGE_ACCOUNT: Optional[str] = os.getenv('AZURE_STORAGE_ACCOUNT')
    AZURE_STORAGE_KEY: Optional[str] = os.getenv('AZURE_STORAGE_KEY')
    
    # API configuration
    API_HOST: str = os.getenv('API_HOST', '0.0.0.0')
    API_PORT: int = int(os.getenv('API_PORT', '8000'))
    DEBUG: bool = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # Model configuration
    MODEL_PATH: str = os.getenv('MODEL_PATH', 'STOUT-V2/models')
    GPU_ENABLED: bool = os.getenv('GPU_ENABLED', 'True').lower() == 'true'
    
    # Evaluation configuration
    MAX_BATCH_SIZE: int = int(os.getenv('MAX_BATCH_SIZE', '1000'))
    EVALUATION_OUTPUT_DIR: str = os.getenv('EVALUATION_OUTPUT_DIR', 'evaluation_results')

def get_azure_connection_string() -> Optional[str]:
    """Get Azure connection string from environment"""
    return Settings.AZURE_CONNECTION_STRING

def get_settings() -> Settings:
    """Get application settings"""
    return Settings()

# Global settings instance
settings = get_settings() 