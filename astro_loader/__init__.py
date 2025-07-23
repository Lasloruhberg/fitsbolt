# Core imports
from .image_loader import load_and_process_images, process_image, SUPPORTED_IMAGE_EXTENSIONS

# Import from submodules
from .normalisation.NormalisationMethod import NormalisationMethod
from .normalisation.normalisation import normalise_image
from .cfg.create_config import create_config, validate_config

__version__ = "0.1.0"

__all__ = [
    # Main functionality
    "load_and_process_images",
    "process_image",
    "SUPPORTED_IMAGE_EXTENSIONS",
    # Normalisation module
    "NormalisationMethod",
    "normalise_image",
    # Configuration module
    "create_config",
    "validate_config",
]
