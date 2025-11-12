# fitsbolt - A Python package for image loading and processing
# Copyright (C) <2025>  <Ruhberg>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the MIT License

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from .normalisation.NormalisationMethod import NormalisationMethod
from .normalisation.normalisation import _normalise_image
from .cfg.create_config import create_config, validate_config, recompute_config_channel_combination
from .cfg.logger import logger
from .resize import _resize_image
from .read import _read_image
from .channel_mixing import batch_channel_combination


def _process_image(
    image,
    cfg,
    image_source="array",
):
    """
    Process an image array by resizing, combining channels, and normalising it.
    Args:
        image (numpy.ndarray): Image array to process H,W or H,W,C
        cfg: Configuration object containing size, normalisation_method
        image_source (str): Source of the image for logging
    Returns:
        numpy.ndarray: Processed image array in specified output dtype (H,W or H,W,C)
    """
    try:
        logger.trace("Processing image with order: resize → combine channels → normalise")
        # Expect a H,W,C image
        im_is_2d = False
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            im_is_2d = True
        # if type is conversion only, we do *not* want to give floats to normalise!
        if cfg.normalisation_method == NormalisationMethod.CONVERSION_ONLY:
            # get dtype of image
            do_type_conversion = True
            combination_output_dtype = image.dtype
        else:
            do_type_conversion = False
            combination_output_dtype = None
        # Step 1: Resize (keep as float for processing chain)
        # the output_dtype ensures that the image dtype is kept for normalise
        image = _resize_image(
            image, cfg, output_dtype=combination_output_dtype, do_type_conversion=do_type_conversion
        )  # type conversion here

        # Step 2: Combine channels before normalisation
        channel_combination_exists = cfg.get("channel_combination") is not None
        if not channel_combination_exists:
            recompute_config_channel_combination(cfg)

        # batch expects a 1, H,W,C image
        image = np.expand_dims(image, axis=0)
        image = batch_channel_combination(
            image,
            cfg.channel_combination,
            output_dtype=combination_output_dtype,  # Keep as float for normalization
        )
        # want to squeeze the image axis again
        image = np.squeeze(image, axis=0)

        # Step 3: Normalize after channel combination
        image = _normalise_image(image, cfg=cfg)

        if im_is_2d:
            image = np.squeeze(image, axis=-1)
            # if image was 2D, we want to return it like this

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
    norm_crop_for_maximum_value=None,
    norm_log_calculate_minimum_value=False,
    norm_log_scale_a=1000.0,
    norm_asinh_scale=[0.7],
    norm_asinh_clip=[99.8],
    norm_zscale_n_samples=1000,
    norm_zscale_contrast=0.25,
    norm_zscale_max_reject=0.5,
    norm_zscale_min_pixels=5,
    norm_zscale_krej=2.5,
    norm_zscale_max_iter=5,
    norm_midtones_percentile=99.8,
    norm_midtones_desired_mean=0.2,
    norm_midtones_crop=None,
    desc="Loading images",
    show_progress=True,
    log_level="WARNING",
    cfg=None,
):
    """Load and process multiple images in parallel.
        this will first read the image, then resize it, then normalise it and finally combine channels.

    Args:
        filepaths (list): filepath or list of image filepaths to load, or list of lists for multi-FITS mode
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
        norm_maximum_value (float, optional): Maximum value for normalisation. Defaults to None implying dynamic.
        norm_minimum_value (float, optional): Minimum value for normalisation. Defaults to None implying dynamic.
        norm_crop_for_maximum_value (tuple, optional): Crops the image to a size of (h,w) around the center to compute
                                    the maximum value inside. Defaults to None.
        Default Log settings
            norm_log_calculate_minimum_value (bool, optional): If True, calculates the minimum value for log scaling.
                                Defaults to False.
            norm_log_scale_a (float, optional): Scale factor for astropy log_stretch. Defaults to 1000.0.
        Default Asinh settings
            norm_asinh_scale (list, optional): Scale factors for asinh normalisation,
                                                should have the length of n_output_channels or 1. Defaults to [0.7].
            norm_asinh_clip (list, optional): Clip values for asinh normalisation,
                                                should have the length of n_output_channels or 1. Defaults to [99.8].
        Default ZScale settings (from astropy ZScaleInterval):
            norm_zscale_n_samples (int, optional): Number of samples for zscale normalisation. Defaults to 1000.
            norm_zscale_contrast (float, optional): Contrast for zscale normalisation. Defaults to 0.25.
            norm_zscale_max_reject (float, optional): Maximum rejection fraction for zscale normalisation. Defaults to 0.5.
            norm_zscale_min_pixels (int, optional): Minimum number of pixels that must remain after rejection
                                                    for zscale normalisation. Defaults to 5.
            norm_zscale_krej (float, optional): The number of sigma used for the rejection. Defaults to 2.5.
            norm_zscale_max_iter (int, optional): Maximum number of iterations for zscale normalisation. Defaults to 5.

        Default MTF settings:
            norm_midtones_percentile (float, optional): Percentile for MTF applied to each channel, in ]0., 100.].
                                                        Defaults to 99.8.
            norm_midtones_desired_mean (float, optional): Desired mean for MTF, in [0, 1]. Defaults to 0.2.
            norm_midtones_crop (tuple, optional): Crops the image to a size of (h,w) around the center to determine the mean in
                                                  Defaults to None.

        desc (str): Description for the progress bar
        show_progress (bool): Whether to show a progress bar
        log_level (str, optional): Logging level for the operation. Defaults to "WARNING".
                                   Can be "TRACE", "DEBUG", "INFO", "WARNING", "ERROR", or "CRITICAL".
        cfg (DotMap, optional): Configuration settings. Defaults to None.


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
            norm_log_scale_a=norm_log_scale_a,
            norm_crop_for_maximum_value=norm_crop_for_maximum_value,
            norm_asinh_scale=norm_asinh_scale,
            norm_asinh_clip=norm_asinh_clip,
            norm_zscale_n_samples=norm_zscale_n_samples,
            norm_zscale_contrast=norm_zscale_contrast,
            norm_zscale_max_reject=norm_zscale_max_reject,
            norm_zscale_min_pixels=norm_zscale_min_pixels,
            norm_zscale_krej=norm_zscale_krej,
            norm_zscale_max_iter=norm_zscale_max_iter,
            norm_midtones_percentile=norm_midtones_percentile,
            norm_midtones_desired_mean=norm_midtones_desired_mean,
            norm_midtones_crop=norm_midtones_crop,
            log_level=log_level,
        )
    else:
        validate_config(cfg)

    # Add a new logger configuration for console output
    logger.set_log_level(cfg.log_level)

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
            if image is None:
                logger.error(f"Failed to load image from {filepath}")
                raise ValueError(
                    f"Image loading failed for {filepath}. Check the file format and content."
                )
            return image
        except Exception as e:
            logger.error(f"Error loading {filepath}: {str(e)}")
            raise e

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
