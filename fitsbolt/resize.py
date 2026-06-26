# fitsbolt - A Python package for image loading and processing
# Copyright (C) <2025>  <Ruhberg>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the MIT or GPL-3.0 License

import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

from .cfg.create_config import create_config
from .cfg.logger import logger
import cv2

# Maps the `interpolation` config value to the cv2 upscaling kernel flag.
# Downscaling always uses cv2.INTER_AREA regardless of this setting.
_CV2_UPSCALE_INTERPOLATION = {
    0: cv2.INTER_NEAREST,
    1: cv2.INTER_LINEAR,
    2: cv2.INTER_CUBIC,
    3: cv2.INTER_LANCZOS4,
    4: cv2.INTER_AREA,  # always used for downscaling
}


def _worker_resize_image(image, cfg):
    """Module-level worker for ProcessPoolExecutor compatibility."""
    try:
        return _resize_image(image, cfg)
    except Exception as e:
        logger.error(f"Error resizing image: {str(e)}")
        raise e


def resize_images(
    images,
    output_dtype=np.uint8,
    size=None,
    interpolation_order=1,
    num_workers=4,
    desc="Resizing images",
    show_progress=True,
    log_level="WARNING",
    use_multiprocessing=False,
):
    """
    Resize an image to the specified size using opencv2's resize function.
    Arrays are cast into float 32, resized and converted

    Args:
        images (list(numpy.ndarray)): List of image arrays to resize
        output_dtype (type, optional): Desired output data type for the resized images.
                                       Can be np.unit8, np.uint16 or np.float32. Defaults to np.uint8.
        size (tuple, optional): Target size for resizing (height, width). If None, no resizing is done.
        interpolation_order (int, optional): Order of interpolation for resizing with opencv2, 0-4. Defaults to 1 = linear
                                             downscaling always uses cv2.INTER_AREA = 4 regardless of this setting.
                                             0= nearest, 1= linear , 2=cubic, 3= lanczos4, 4= inter_area
        log_level (str, optional): Logging level for the operation. Defaults to "WARNING".
                                   Can be "TRACE", "DEBUG", "INFO", "WARNING", "ERROR", or "CRITICAL".
        use_multiprocessing (bool, optional): Use ProcessPoolExecutor instead of ThreadPoolExecutor.
                                              Bypasses GIL for CPU-heavy workloads. Defaults to False.
    Returns:
        list(numpy.ndarray): List of resized image arrays

    """
    cfg = create_config(
        output_dtype=output_dtype,
        size=size,
        interpolation_order=interpolation_order,
        num_workers=num_workers,
        log_level=log_level,
    )
    # Add a new logger configuration for console output
    logger.set_log_level(cfg.log_level)

    logger.debug(f"Setting LogLevel to {cfg.log_level.upper()}")

    logger.debug(
        f"Loading {len(images)} images in parallel with normalisation: {cfg.normalisation_method}"
    )

    worker_fn = partial(_worker_resize_image, cfg=cfg)
    Executor = ProcessPoolExecutor if use_multiprocessing else ThreadPoolExecutor

    # Use executor for parallel loading
    with Executor(max_workers=cfg.num_workers) as executor:
        if show_progress:
            results = list(
                tqdm(
                    executor.map(worker_fn, images),
                    desc=desc,
                    total=len(images),
                )
            )
        else:
            results = list(executor.map(worker_fn, images))

    logger.debug(f"Successfully loaded {len(results)} of {len(images)} images")
    return results


def resize_image(
    image, output_dtype=np.uint8, size=None, interpolation_order=1, log_level="WARNING"
):
    """
    Resize an image to the specified size using opencv2's resize function.
    Arrays are cast into float 32, resized and converted

    Args:
        image (numpy.ndarray): Image array to resize
        output_dtype (type, optional): Desired output data type for the resized images.
                                       Can be np.unit8, np.uint16 or np.float32. Defaults to np.uint8.
        size (tuple, optional): Target size for resizing (height, width). If None, no resizing is done.
        interpolation_order (int, optional): Order of interpolation for resizing with opencv2, 0-4. Defaults to 1 = linear
                                             downscaling always uses cv2.INTER_AREA = 4 regardless of this setting.
                                             0= nearest, 1= linear , 2=cubic, 3= lanczos4, 4= inter_area
        log_level (str, optional): Logging level for the operation. Defaults to "WARNING".
                                   Can be "TRACE", "DEBUG", "INFO", "WARNING", "ERROR", or "CRITICAL".
    Returns:
        numpy.ndarray: Resized image array
    """
    cfg = create_config(
        output_dtype=output_dtype,
        size=size,
        interpolation_order=interpolation_order,
        log_level=log_level,
    )

    return _resize_image(image, cfg)


def _resize_image(image, cfg, output_dtype=None, do_type_conversion=True):
    """Resize an image to the specified size using opencv2's resize function.
    Arrays are cast into float 32, resized and converted

    Args:
        image (np.ndarray): Image array to resize
        cfg (Dict): Configuration dictionary containing resize parameters
        output_dtype (type, optional): Desired output data type for the resized image. Defaults to None.
                                        Overwrites the cfg parameter if set (used in image_loader.py's _process_image)
        do_type_conversion (bool, optional): Whether to perform type conversion. Defaults to True.

    Raises:
        ValueError: If the image is empty.
        ValueError: If the image cannot be resized.
        ValueError: If the output dtype is not supported.

    Returns:
        np.ndarray: Resized image array
    """
    # Simple resize that maintains uint8 type if requested
    if image.size == 0:
        logger.warning("Received an empty image, returning as is.")
        raise ValueError("Image is empty, cannot resize.")
    if cfg.size is not None and image.shape[:2] != tuple(cfg.size):
        resize_func = cv2.resize

        upscale_interpolation = (
            _CV2_UPSCALE_INTERPOLATION[cfg.interpolation_order]
            if cfg.interpolation_order is not None
            else cv2.INTER_LINEAR
        )

        downscaling = cfg.size[0] < image.shape[0] or cfg.size[1] < image.shape[1]
        interpolation_flag = cv2.INTER_AREA if downscaling else upscale_interpolation
        # cv2 drops channel dimension for single-channel images, so we need to add it back
        readd_channel = False
        if image.ndim == 3 and image.shape[2] == 1:
            readd_channel = True

        # cv2 internally returns the input dtype
        image = resize_func(
            np.ascontiguousarray(image, dtype=np.float32),
            cfg.size,
            interpolation=interpolation_flag,
        )
        if readd_channel:
            image = np.expand_dims(image, axis=-1)
        if do_type_conversion and (cfg.output_dtype is not None or output_dtype is not None):
            if output_dtype:
                target = output_dtype
            else:
                target = cfg.output_dtype

            # the resizing creates floats, so proper clipping and conversion is needed
            if target == np.uint8:
                image = np.clip(image, 0, np.iinfo(np.uint8).max).astype(np.uint8)
            elif target == np.uint16:
                image = np.clip(image, 0, np.iinfo(np.uint16).max).astype(np.uint16)
            elif image.dtype != target:
                image = image.astype(target)
    if image is None:
        logger.error("Failed to resize image")
        raise ValueError("Image resizing failed. Check the file format and content.")
    if isinstance(image, np.ndarray) and image.size == 0:
        logger.warning("Received an empty image, returning as is.")
        raise ValueError("Image resizing failed.")
    return image
