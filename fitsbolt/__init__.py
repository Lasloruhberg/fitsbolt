# fitsbolt - A Python package for image loading and processing
# Copyright (C) <2025>  <Ruhberg>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the MIT License

# Core imports
from .image_loader import load_and_process_images
from .read import read_images
from .resize import resize_images

# Import from submodules
from .normalisation.NormalisationMethod import NormalisationMethod
from .normalisation.normalisation import normalise_images
from .cfg.create_config import create_config, validate_config, SUPPORTED_IMAGE_EXTENSIONS
from .channel_mixing import batch_channel_combination

__version__ = "0.1.4"

__all__ = [
    # Main functionality
    "load_and_process_images",
    "SUPPORTED_IMAGE_EXTENSIONS",
    # Individual processing functions
    "read_images",
    "normalise_images",
    "resize_images",
    "batch_channel_combination",
    # Normalisation module
    "NormalisationMethod",
    "normalise_image",
    # Configuration module
    "create_config",
    "validate_config",
]
