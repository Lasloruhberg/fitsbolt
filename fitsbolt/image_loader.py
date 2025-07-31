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

import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from loguru import logger

from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod
from fitsbolt.normalisation.normalisation import _normalise_image
from fitsbolt.cfg.create_config import create_config, validate_config
from fitsbolt.resize import _resize_image
from fitsbolt.read import _read_image


def _process_image(
    image,
    cfg,
    image_source="array",
):
    """
    Process an image array by normalising and resizing it.
    Args:
        image (numpy.ndarray): Image array to process
        cfg: Configuration object containing size, normalisation_method
        image_source (str): Source of the image for logging
    Returns:
        numpy.ndarray: Processed image array as uint8
    """
    try:
        logger.trace(f"Normalising image with setting {cfg.normalisation_method}")
        image = _normalise_image(image, cfg=cfg)

        image = _resize_image(image, cfg)

        return image

    except Exception as e:
        logger.error(f"Error processing image {image_source}: {e}")
        raise e


def _load_image(filepath, cfg):
    try:
        # Read raw image data
        image = _read_image(filepath, cfg)

        # Process the image using the centralized processing function
        return _process_image(image, cfg, image_source=filepath)

    except Exception as e:
        logger.error(f"Error reading image {filepath}: {e}")
        raise e


def load_and_process_images(
    filepaths,
    cfg=None,
    output_dtype=np.uint8,
    size=[224, 224],
    fits_extension=None,
    interpolation_order=1,
    normalisation_method=NormalisationMethod.CONVERSION_ONLY,
    channel_combination=None,
    n_output_channels=3,
    num_workers=4,
    norm_maximum_value=None,
    norm_minimum_value=None,
    norm_log_calculate_minimum_value=False,
    norm_crop_for_maximum_value=None,
    norm_asinh_scale=[0.7, 0.7, 0.7],
    norm_asinh_clip=[99.8, 99.8, 99.8],
    desc="Loading images",
    show_progress=True,
):
    """Load and process multiple images in parallel.
        this will first read the image, then normalise it and finally resize it.

    Args:
        filepaths (list): filepath or list of image filepaths to load, or list of lists for multi-FITS mode
        cfg (DotMap, optional): Configuration settings. Defaults to None.
        output_dtype (type, optional): Data type for output images. Defaults to np.uint8.
        size (list, optional): Target size for image resizing. Defaults to [224, 224].
        fits_extension (int, str, list, optional): The FITS extension(s) to use. Can be:
                                               - An integer index
                                               - A string extension name
                                               - A list of integers or strings to combine multiple extensions
                                               - For multi-FITS mode: list of extensions matching filepaths structure
                                               Uses the first extension (0) if None.
        interpolation_order (int, optional): Order of interpolation for resizing with skimage, 0-5. Defaults to 1.
        normalisation_method (NormalisationMethod, optional): Normalisation method to use.
                                                Defaults to NormalisationMethod.CONVERSION_ONLY.
        channel_combination (dict, optional): Dictionary defining how to combine FITS extensions into output channels.
                                                Defaults to None.
        n_output_channels (int, optional): Number of output channels for the image. Defaults to 3.
        num_workers (int, optional): Number of worker threads for data loading. Defaults to 4.
        norm_maximum_value (float, optional): Maximum value for normalisation. Defaults to None.
        norm_minimum_value (float, optional): Minimum value for normalisation. Defaults to None.
        norm_log_calculate_minimum_value (bool, optional): If True, calculates the minimum value when log scaling
                                                (normally defaults to 0). Defaults to False.
        norm_crop_for_maximum_value (tuple, optional): Crops the image for maximum value. Defaults to None.
        norm_asinh_scale (list, optional): Scale factors for asinh normalisation. Defaults to [0.7, 0.7, 0.7].
        norm_asinh_clip (list, optional): Clip values for asinh normalisation. Defaults to [99.8, 99.8, 99.8].
        filepaths (list): List of image filepaths to load
        size (tuple, optional): Size to resize images to (height, width)
        desc (str): Description for the progress bar
        show_progress (bool): Whether to show a progress bar

    Returns:
        list: List of images for successfully loaded and processed images

    Examples:
        # Single FITS file with multiple extensions
        images = load_and_process_images(["image.fits"], fits_extension=[0, 1, 2])

        # Multiple FITS files with corresponding extensions (new functionality)
        images = load_and_process_images([["file1.fits", "file2.fits", "file3.fits"]],
                                       fits_extension=[0, 1, 2])
    """
    # check if input is a single filepath or a list
    if not isinstance(filepaths, (list, np.ndarray)):
        return_single = True
        filepaths = [filepaths]
    else:
        return_single = False

    # Check for multi-FITS mode (nested lists)
    if filepaths and isinstance(filepaths[0], list):
        # Multi-FITS mode: each element is a list of FITS files
        logger.debug("Multi-FITS mode detected: combining multiple FITS files per image")
        if not isinstance(fits_extension, list):
            raise ValueError(
                "Multi-FITS mode requires fits_extension to be a list matching the number of files"
            )

    if cfg is None:
        cfg = create_config(
            output_dtype=output_dtype,
            size=size,
            fits_extension=fits_extension,
            interpolation_order=interpolation_order,
            normalisation_method=normalisation_method,
            channel_combination=channel_combination,
            n_output_channels=n_output_channels,
            num_workers=num_workers,
            norm_maximum_value=norm_maximum_value,
            norm_minimum_value=norm_minimum_value,
            norm_log_calculate_minimum_value=norm_log_calculate_minimum_value,
            norm_crop_for_maximum_value=norm_crop_for_maximum_value,
            norm_asinh_scale=norm_asinh_scale,
            norm_asinh_clip=norm_asinh_clip,
        )
    else:
        validate_config(cfg)
    # initialise logger
    logger.remove()

    # Add a new logger configuration for console output
    logger.add(
        sys.stderr,
        colorize=True,
        level=cfg.log_level.upper(),
        format="<green>{time:HH:mm:ss}</green>|astro-loader-<blue>{level}</blue>| <level>{message}</level>",
    )

    logger.debug(f"Setting LogLevel to {cfg.log_level.upper()}")

    logger.debug(
        f"Loading {len(filepaths)} images in parallel with normalisation: {cfg.normalisation_method}"
    )

    def load_single_image(filepath):
        try:
            image = _load_image(
                filepath,
                cfg,
            )
            return image
        except Exception as e:
            logger.error(f"Error loading {filepath}: {str(e)}")
            return None

    # Use ThreadPoolExecutor for parallel loading
    with ThreadPoolExecutor(max_workers=cfg.num_workers) as executor:
        if show_progress:
            results = list(
                tqdm(
                    executor.map(load_single_image, filepaths),
                    desc=desc,
                    total=len(filepaths),
                )
            )
        else:
            results = list(executor.map(load_single_image, filepaths))

    # Filter out None results (failed loads)
    results = [r for r in results if r is not None]

    logger.debug(f"Successfully loaded {len(results)} of {len(filepaths)} images")
    if return_single:
        # If only one image was requested, return it directly
        if len(results) == 1:
            return results[0]
        else:
            logger.warning(
                "Multiple images loaded but only one was requested. Returning the first image."
            )
            return results[0]
    return results
