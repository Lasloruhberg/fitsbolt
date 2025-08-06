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
from skimage.transform import resize
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


from fitsbolt.cfg.create_config import create_config


def resize_images(
    images,
    output_dtype=np.uint8,
    size=None,
    interpolation_order=1,
    num_workers=4,
    desc="Resizing images",
    show_progress=True,
):
    """
    Resize an image to the specified size using skimage's resize function.

    Args:
        images (list(numpy.ndarray)): List of image arrays to resize
        size (tuple, optional): Target size for resizing (height, width). If None, no resizing is done.
        interpolation_order (int, optional): Order of interpolation for resizing with skimage, 0-5. Defaults to 1.

    Returns:
        list(numpy.ndarray): List of resized image arrays

    """
    cfg = create_config(
        output_dtype=output_dtype,
        size=size,
        interpolation_order=interpolation_order,
        num_workers=num_workers,
    )

    # initialise logger
    logger.remove()

    # Add a new logger configuration for console output
    logger.add(
        sys.stderr,
        colorize=True,
        level=cfg.log_level.upper(),
        format="<green>{time:HH:mm:ss}</green>|fitsbolt-<blue>{level}</blue>| <level>{message}</level>",
    )

    logger.debug(f"Setting LogLevel to {cfg.log_level.upper()}")

    logger.debug(
        f"Loading {len(images)} images in parallel with normalisation: {cfg.normalisation_method}"
    )

    def resize_single_image(image):
        try:
            image = _resize_image(
                image,
                cfg,
            )
            return image
        except Exception as e:
            logger.error(f"Error resizing {image}: {str(e)}")
            raise e

    # Use ThreadPoolExecutor for parallel loading
    with ThreadPoolExecutor(max_workers=cfg.num_workers) as executor:
        if show_progress:
            results = list(
                tqdm(
                    executor.map(resize_single_image, images),
                    desc=desc,
                    total=len(images),
                )
            )
        else:
            results = list(executor.map(resize_single_image, images))

    logger.debug(f"Successfully loaded {len(results)} of {len(images)} images")
    return results


def resize_image(image, output_dtype=np.uint8, size=None, interpolation_order=1):
    """
    Resize an image to the specified size using skimage's resize function.

    Args:
        image (numpy.ndarray): Image array to resize
        size (tuple, optional): Target size for resizing (height, width). If None, no resizing is done.
        interpolation_order (int, optional): Order of interpolation for resizing with skimage, 0-5. Defaults to 1.

    Returns:
        numpy.ndarray: Resized image array
    """
    cfg = create_config(
        output_dtype=output_dtype,
        size=size,
        interpolation_order=interpolation_order,
    )

    return _resize_image(image, cfg)


def _resize_image(image, cfg):
    # Simple resize that maintains uint8 type if requested
    if image.size == 0:
        logger.warning("Received an empty image, returning as is.")
        return image
    if cfg.size is not None and image.shape[:2] != tuple(cfg.size):
        image = resize(
            image,
            cfg.size,
            anti_aliasing=None,
            order=cfg.interpolation_order if cfg.interpolation_order is not None else 1,
            preserve_range=True,
        )
        # the resizing creates floats, so proper clipping and conversion is needed
        if cfg.output_dtype == np.uint8:
            image = np.clip(image, 0, np.iinfo(np.uint8).max).astype(np.uint8)
        elif cfg.output_dtype == np.uint16:
            image = np.clip(image, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        elif image.dtype != cfg.output_dtype:
            image = image.astype(cfg.output_dtype)
    if image is None:
        logger.error("Failed to resize image")
        raise ValueError("Image resizing failed. Check the file format and content.")
    if isinstance(image, np.ndarray) and image.size == 0:
        logger.warning("Received an empty image, returning as is.")
        raise ValueError("Image resizing failed.")
    return image
