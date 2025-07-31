# fitsbolt - A Python package for image loading and processing
# Copyright (C) <2025>  <Ruhberg>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Core imports
from .image_loader import load_and_process_images
from .read import read_images
from .resize import resize_images

# Import from submodules
from .normalisation.NormalisationMethod import NormalisationMethod
from .normalisation.normalisation import normalise_images
from .cfg.create_config import create_config, validate_config, SUPPORTED_IMAGE_EXTENSIONS

__version__ = "0.1.0.dev3"

__all__ = [
    # Main functionality
    "load_and_process_images",
    "SUPPORTED_IMAGE_EXTENSIONS",
    # Individual processing functions
    "read_images",
    "normalise_images",
    "resize_images",
    # Normalisation module
    "NormalisationMethod",
    "normalise_image",
    # Configuration module
    "create_config",
    "validate_config",
]
