"""
Environment configuration utilities for dataset paths.
"""
import os
from pathlib import Path
from dotenv import load_dotenv


def load_dataset_config():
    """
    Load dataset configuration from .env file.
    
    Returns:
        dict: Dictionary containing dataset paths
    """
    # Load .env file from the project root
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"
    
    if env_path.exists():
        load_dotenv(env_path)
    else:
        print(f"Warning: .env file not found at {env_path}")
        print("Please copy .env.example to .env and update the paths")
    
    # Get environment variables with fallbacks
    config = {
        'dataset_root': os.getenv('DATASET_ROOT', '/path/to/your/datasets/New_SwallowSet'),
        'raw_data_dir': os.getenv('RAW_DATA_DIR', '/path/to/your/datasets/New_SwallowSet/Raw'),
        'short_audio_dir': os.getenv('SHORT_AUDIO_DIR', '/path/to/your/datasets/New_SwallowSet/Test'),
        'long_audio_dir': os.getenv('LONG_AUDIO_DIR', '/path/to/your/datasets/New_SwallowSet/Long'),
    }
    
    return config


def get_dataset_root():
    """Get the root dataset directory."""
    config = load_dataset_config()
    return config['dataset_root']


def get_raw_data_dir():
    """Get the raw data directory."""
    config = load_dataset_config()
    return config['raw_data_dir']


def get_short_audio_dir():
    """Get the short audio segments directory."""
    config = load_dataset_config()
    return config['short_audio_dir']


def get_long_audio_dir():
    """Get the long audio files directory."""
    config = load_dataset_config()
    return config['long_audio_dir']