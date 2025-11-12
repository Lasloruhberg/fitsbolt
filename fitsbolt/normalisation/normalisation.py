# fitsbolt - A Python package for image loading and processing
# Copyright (C) <2025>  <Ruhberg>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the MIT License

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import warnings

from skimage.util import img_as_ubyte, img_as_uint, img_as_float32

from astropy.visualization import (
    ImageNormalize,
    LogStretch,
    LinearStretch,
    ZScaleInterval,
    AsinhStretch,
    PercentileInterval,
)

from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod
from fitsbolt.cfg.create_config import create_config
from fitsbolt.cfg.logger import logger


def _type_conversion(data: np.ndarray, cfg) -> np.ndarray:
    """Convert the image data to the specified output dtype."""
    if cfg.output_dtype == np.uint8:
        return img_as_ubyte(data)
    elif cfg.output_dtype == np.uint16:
        return img_as_uint(data)
    elif cfg.output_dtype == np.float32:
        return img_as_float32(data)
    else:
        # Default to uint8 if output_dtype is not specified or not supported
        warnings.warn(f"Unsupported output dtype: {cfg.output_dtype}, defaulting to uint8")
        return img_as_ubyte(data)


def _crop_center(data: np.ndarray, crop_height: int, crop_width: int) -> np.ndarray:
    """
    Crop the central region of an image.

    Parameters:
    - data: np.ndarray
        Input image as (H, W, ...) array.
    - crop_height: int
        Height of the cropped region.
    - crop_width: int
        Width of the cropped region.

    Returns:
    - np.ndarray
        Cropped central region.
    """
    h, w = data.shape[:2]
    top = (h - crop_height) // 2
    left = (w - crop_width) // 2
    if top < 0 or left < 0:
        warnings.warn("Crop size is larger than image size, returning original image")
        return data
    return data[top : top + crop_height, left : left + crop_width]


def _compute_max_value(data, cfg=None):
    """Compute the maximum value of the image for normalisation
    Args:
        data (numpy array): Input image array, can be high dynamic range
        cfg (DotMap or None): Configuration with optional normalisation values.
    Returns:
        float: Maximum value for normalisation
    """

    if (
        cfg.normalisation.crop_for_maximum_value is not None
        and cfg.normalisation.maximum_value is None
    ):
        h, w = cfg.normalisation.crop_for_maximum_value
        assert (
            h > 0 and w > 0
        ), f"Crop size must be positive integers currently {cfg.normalisation.crop_for_maximum_value}"
        # make cutout of the image and compute max value
        img_centre_region = _crop_center(data, h, w)
        max_value = np.nanmax(img_centre_region)

    else:
        # Compute the maximum value of the image
        max_value = (
            cfg.normalisation.maximum_value
            if cfg.normalisation.maximum_value is not None
            else np.nanmax(data)
        )

    return max_value


def _compute_min_value(data, cfg):
    """Compute the minimum value of the image for normalisation
    Args:
        data (numpy array): Input image array, can be high dynamic range
        cfg (DotMap or None): Configuration with optional normalisation values.
    Returns:
        float: Maximum value for normalisation
    """
    min_value = (
        cfg.normalisation.minimum_value
        if cfg.normalisation.minimum_value is not None
        else np.nanmin(data)
    )

    return min_value


def _log_normalisation(data, cfg):
    """A log normalisation based on a minimum as 0 (bkg subtracted) or higher (if calc_vmin is True)
    and a dynamically determined maximum. If cfg.normalisation.crop_for_maximum_value is not None the maximum is determined
    on a crop around the center, with the shape given by the Tuple crop_for_maximum_value.

    Args:
        data (numpy array): Input image array, ideally a float32 or float64 array, can be high dynamic range
        cfg (DotMap or None): Configuration with optional normalisation values.
            cfg.normalisation.log_calculate_minimum_value (bool): If True, calculate the minimum value of the image,
            otherwise set to 0 or cfg.normalisation.minimum_value if set
            cfg.normalisation.crop_for_maximum_value (Tuple[int, int], optional): Width and height to crop around the center,
            to calculate the maximum value in
            cfg.normalisation.log_scale_a (float): a parameter of astropys log stretch, default 1000.0
            cfg.output_dtype: The desired output data type

    Returns:
        numpy array: A normalised image in the specified output data type
    """

    if cfg.normalisation.log_calculate_minimum_value:
        minimum = _compute_min_value(data, cfg=cfg)
    else:
        minimum = (
            cfg.normalisation.minimum_value if cfg.normalisation.minimum_value is not None else 0.0
        )

    maximum = _compute_max_value(data, cfg=cfg)
    if minimum < maximum:
        norm = ImageNormalize(
            data,
            vmin=minimum,
            vmax=maximum,
            stretch=LogStretch(a=cfg.normalisation.log_scale_a),
            clip=True,
        )
    else:
        warnings.warn("Image maximum is not larger than minimum, using linear normalisation")
        norm = ImageNormalize(
            data,
            vmin=None,
            vmax=None,
            stretch=LogStretch(a=cfg.normalisation.log_scale_a),
            clip=True,
        )
    img_normalised = norm(data)  # range 0,1
    # Convert back to uint8 range
    return _type_conversion(img_normalised, cfg)


def _linear_normalisation(data, cfg):
    """A linear normalisation

    Args:
        data (numpy array): Input image array, ideally a float32 or float64 array, can be high dynamic range
        cfg (DotMap or None): Configuration with optional normalisation values.
            cfg.normalisation.log_calculate_minimum_value (bool): If True, calculate the minimum value of the image,
            otherwise set to 0 or cfg.normalisation.minimum_value if set
            cfg.normalisation.crop_for_maximum_value (Tuple[int, int], optional): Width and height to crop around the center,
            to calculate the maximum value in
            cfg.output_dtype: The desired output data type

    Returns:
        numpy array: A normalised image in the specified output data type
    """

    minimum = _compute_min_value(data, cfg=cfg)
    maximum = _compute_max_value(data, cfg=cfg)
    if minimum < maximum:
        norm = ImageNormalize(data, vmin=minimum, vmax=maximum, stretch=LinearStretch(), clip=True)
    else:
        warnings.warn(
            "Image maximum is not larger than minimum, only doing conversion normalisation"
        )
        return _conversiononly_normalisation(data, cfg)
    img_normalised = norm(data)  # range 0,1
    # Convert back to type range
    return _type_conversion(img_normalised, cfg)


def _zscale_normalisation(data, cfg):
    """A linear zscale normalisation

    Args:
        data (numpy array): Input image array, ideally a float32 or float64 array
        cfg (DotMap): Configuration with normalisation values and output dtype

    Returns:
        numpy array: A normalised image in the specified output data type
    """
    if not np.any(data != data.flat[0]):  # Constant value check
        warnings.warn("Zscale normalisation: constant image detected, using fallback conversion.")
        return _conversiononly_normalisation(data, cfg)

    # Min Max value do not apply, also no constrain to center
    norm = ImageNormalize(
        data,
        interval=ZScaleInterval(
            n_samples=cfg.normalisation.zscale.n_samples,
            contrast=cfg.normalisation.zscale.contrast,
            max_reject=cfg.normalisation.zscale.max_reject,
            min_npixels=cfg.normalisation.zscale.min_npixels,
            krej=cfg.normalisation.zscale.krej,
            max_iterations=cfg.normalisation.zscale.max_iterations,
        ),
        stretch=LinearStretch(),
        clip=True,
    )
    img_normalised = norm(data)  # range 0,1
    if np.max(img_normalised) > np.min(img_normalised):
        # Convert back to specified dtype
        return _type_conversion(img_normalised, cfg)
    else:
        warnings.warn(
            "Zscale normalisation: image maximum value not larger than minimum, only converting image"
        )
        return _conversiononly_normalisation(data, cfg)


def _conversiononly_normalisation(data, cfg):
    """A normalisation that does not change the image, but only converts it to the specified dtype

    Args:
        data (numpy array): Input image array, can have a high dynamic range
        cfg (DotMap): Configuration with optional normalisation values.
            cfg.normalisation.crop_for_maximum_value (Tuple[int, int], optional): Width and height to crop around the center,
            to compute the maximum value in
            cfg.output_dtype: The desired output data type (np.uint8, np.uint16, np.float32)

    Returns:
        numpy array: A converted image in the specified output dtype any float output will be between [0,1]
    """
    # If input dtype already matches the requested output dtype and it's float32,
    # we still need to ensure it's normalised to [0,1] range
    # For any other case, use normalised conversion (e.g. for input floats)

    # get min or max from config if available
    maximum = _compute_max_value(data, cfg)
    minimum = _compute_min_value(data, cfg)
    # clip to cover edge cases
    data = np.clip(data, minimum, maximum)

    if data.dtype == cfg.output_dtype:
        if np.issubdtype(cfg.output_dtype, np.floating):
            # For float output, ensure data is in [0,1] range later on
            pass

        else:
            # For integer dtypes, if they match, return as is
            return data

    # Handle specific direct conversions for better precision
    if cfg.output_dtype == np.uint8:
        if data.dtype == np.uint16:
            # Direct conversion from uint16 to uint8 with proper scaling
            return _type_conversion(data / 65535.0, cfg)  # 65535 = 2^16 - 1
        # if not matching dtype scale to [0,1] and convert
        if maximum > minimum:
            data = (data - minimum) / (maximum - minimum)
        else:
            data = data - minimum  # should return 0
        return _type_conversion(data, cfg)

    elif cfg.output_dtype == np.uint16:
        if data.dtype == np.uint8:
            # Direct conversion from uint8 to uint16 with proper scaling
            return _type_conversion(data / 255.0, cfg)  # Scale to [0,1] then convert

    elif cfg.output_dtype == np.float32:
        if data.dtype == np.uint8:
            # Convert uint8 directly to float32 [0,1] range
            return _type_conversion(data / 255.0, cfg)

        elif data.dtype == np.uint16:
            # Convert uint16 directly to float32 [0,1] range
            return _type_conversion(data / 65535.0, cfg)

    # ensure valid range
    if maximum > minimum:
        norm = ImageNormalize(data, vmin=minimum, vmax=maximum, clip=True)
        img_normalised = norm(data)  # range 0,1
        return _type_conversion(img_normalised, cfg)
    else:
        warnings.warn("Image maximum is not larger than minimum, returning zero array")
        # this is something that can happen with certain settings, so this should not raise an exception
        return np.zeros_like(data, dtype=cfg.output_dtype)


def _expand(value, length: int) -> np.ndarray:
    """Turn a scalar or sequence into a length-`length` float32 array.
    Used in the asinh normalisation to ensure that the scale and clip
    parameters are always arrays of the correct length."""
    if isinstance(value, (list, tuple)):
        arr = np.array(value, dtype=np.float32)
    else:
        arr = np.array([value], dtype=np.float32)
    if arr.size != length:
        # input parameter mismatch
        if arr.size != 1:
            logger.warning(
                f"Parameter norm_asinh_scale or norm_asinh_clip: {value!r} has length {arr.size}, expected {length}."
                + " Will use first element"
            )
        try:
            arr = np.full(length, arr[0], dtype=np.float32)
        except IndexError:
            raise ValueError(f"Cannot shorten {arr!r} to length {length}")
    return arr


def _asinh_normalisation(data, cfg):
    """A normalisation based on the asinh stretch.
    Allows for per-channel scaling and clipping.
    If cfg.normalisation.crop_for_maximum_value is not None the maximum is determined on a cutout around the center

    Args:
    ----------
    data : np.ndarray
        Image array. Either single-channel (any shape) or RGB with
        ``data.ndim == 3`` and ``data.shape[2] == 3``.
    cfg : DotMap
        Configuration object holding
        ``cfg.normalisation.asinh_scale`` and
        ``cfg.normalisation.asinh_clip``.  Each may be a scalar
        or a n(typically 3)-element sequence.
        ``cfg.output_dtype``: The desired output data type.

    Returns
    -------
    np.ndarray
        Asinh-stretched (and possibly clipped) image in the specified output data type.
    """
    # Determine whether we are dealing with RGB+.... or not
    channels = data.shape[-1] if data.ndim == 3 else 1

    # Prepare per-channel parameters
    scale = _expand(cfg.normalisation.asinh_scale, channels)
    clip = _expand(cfg.normalisation.asinh_clip, channels)

    # Get initial min and max and clip values if manual are set
    max_value = _compute_max_value(data, cfg)
    min_value = _compute_min_value(data, cfg)
    data = np.clip(data, min_value, max_value)

    # Apply asinh normalisation & percentile clipping, potentially per-channel
    if channels == 1:
        norm = ImageNormalize(
            data, interval=PercentileInterval(clip[0]), stretch=AsinhStretch(scale[0]), clip=True
        )
        normalised = norm(data)
    else:
        normalised = np.zeros_like(data, dtype=np.float32)
        for c in range(channels):
            # Apply asinh stretch with scale parameter and percentile clipping for each channel
            norm = ImageNormalize(
                data[..., c],
                interval=PercentileInterval(clip[c]),
                stretch=AsinhStretch(scale[c]),
                clip=True,
            )
            normalised[..., c] = norm(data[..., c])
    # correct to 0-1 range and convert to uint8
    min_value = np.min(normalised)
    max_value = np.max(normalised)
    if min_value < max_value:
        return _type_conversion((normalised - min_value) / (max_value - min_value), cfg)
    else:

        warnings.warn("Image maximum is not larger than minimum, returning conversion only.")

        return _conversiononly_normalisation(data, cfg=cfg)


def _apply_midtones_on_normalised_data(x, m):
    """Apply the midtones normalisation

    Args:
        x (np.ndarray): The input image data.
        m (float): The midtones balance parameter.

    Returns:
        np.ndarray: The transformed image data.
    """

    assert x.max() <= 1
    assert x.min() >= 0
    Zero_mask = x == 0
    Midtones_mask = x == m
    Full_mask = x == 1
    mask_else = ~(Zero_mask | Midtones_mask | Full_mask)

    # create an output array that keeps some fixed values
    output = np.zeros_like(x)
    output[Zero_mask] = 0
    output[Midtones_mask] = 0.5
    output[Full_mask] = 1
    x_else = x[mask_else]

    # apply the curve
    output[mask_else] = ((m - 1) * x_else) / ((2 * m - 1) * x_else - m)
    output = np.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0)  # ensure no NaNs or infs
    return output


def _find_mean_of_normalised(normalised_data, cfg):
    """Find the midtones balance parameter m for the given normalised data."""
    if cfg.normalisation.midtones.crop is not None:
        h, w = cfg.normalisation.midtones.crop
        assert (
            h > 0 and w > 0
        ), f"Crop size must be positive integers currently {cfg.normalisation.midtones.crop}"
        # make cutout of the image and compute max value
        normalised_data_cut = _crop_center(normalised_data, h, w)
    else:
        normalised_data_cut = normalised_data
    x = np.mean(normalised_data_cut)
    alpha = cfg.normalisation.midtones.desired_mean
    return (x - alpha * x) / (x - 2 * alpha * x + alpha)


def _midtones_normalisation(data, cfg):
    """Compute the Midtones Transfer Function (MTF) for given x and m.
    This is similar to the "curves" tool from image editing software,
    m sets the curve and MTF is the application of the curve.

    Args:
        x (np.ndarray): The input image data.
        m (float): The midtones balance parameter.

    Returns:
        np.ndarray: The transformed image data.
    """
    # Get initial min and max and clip values if manual are set
    max_value = _compute_max_value(data, cfg)
    min_value = _compute_min_value(data, cfg)
    data = np.clip(data, min_value, max_value)

    data_is_2d = False
    if data.ndim == 2:
        # create dummy channel index
        data_is_2d = True
        data = np.expand_dims(data, axis=-1)
    # create a for loop over the channel to calculate m and apply MTF on a channel basis
    for c in range(data.shape[-1]):
        # do a channel-wise percentile clip
        if cfg.normalisation.midtones.percentile:
            data[..., c] = np.clip(
                data[..., c],
                data[..., c].min(),
                np.percentile(data[..., c], cfg.normalisation.midtones.percentile),
            )
        # Find the appropriate midtones balance parameter m
        max_value = _compute_max_value(data[..., c], cfg)
        min_value = _compute_min_value(data[..., c], cfg)
        # include necessary clipping
        data[..., c] = np.clip(data[..., c], min_value, max_value)
        normalised_channel = (data[..., c] - min_value) / (max_value - min_value)

        m = _find_mean_of_normalised(normalised_channel, cfg)
        # Apply the MTF to the image
        transformed_channel = _apply_midtones_on_normalised_data(normalised_channel, m)

        data[..., c] = transformed_channel
    if data_is_2d:
        data = np.squeeze(data, axis=-1)
    # scale entire image to 0,1 and do type conversion
    max_value = _compute_max_value(data, cfg)
    min_value = _compute_min_value(data, cfg)
    if min_value < max_value:
        return _type_conversion((data - min_value) / (max_value - min_value), cfg)
    else:

        warnings.warn("Image maximum is not larger than minimum, returning conversion only.")

        return _conversiononly_normalisation(data, cfg=cfg)


def _normalise_image(data, cfg):
    """Normalises all images based on the selected normalisation option

    If None is selected and a uint16 array given, it is linearly scaled to uint8
    Otherwise None applies linear normalisation to shift the image to the required [0,255] range if outside of it

    Args:
        data (numpy array): Input image array, can have high dynamic range
        method (NormalisationMethod): Normalisation method enum for test
        cfg (DotMap): Configuration object containing normalisation settings

    Returns:
        numpy array: A normalised image based on the selected method
    """

    # carefully replace nans with 0
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    method = cfg.normalisation_method
    # Method selection
    if isinstance(method, NormalisationMethod):
        pass
    else:
        logger.critical(f"Normalisation method type {method} , {type(method)} not implemented")
        # ensure uint8
        return _conversiononly_normalisation(data, cfg=cfg)

    # execute normalisations based on enum
    if method == NormalisationMethod.LOG:
        return _log_normalisation(data, cfg=cfg)
    if method == NormalisationMethod.LINEAR:
        return _linear_normalisation(data, cfg=cfg)
    elif method == NormalisationMethod.CONVERSION_ONLY:
        return _conversiononly_normalisation(data, cfg=cfg)
    elif method == NormalisationMethod.ZSCALE:
        return _zscale_normalisation(data, cfg=cfg)
    elif method == NormalisationMethod.ASINH:
        return _asinh_normalisation(data, cfg=cfg)
    elif method == NormalisationMethod.MIDTONES:
        return _midtones_normalisation(data, cfg=cfg)
    else:
        logger.critical(f"Normalisation method {method} not implemented")
        return _conversiononly_normalisation(data, cfg=cfg)


def normalise_images(
    images,
    output_dtype=np.uint8,
    normalisation_method=NormalisationMethod.CONVERSION_ONLY,
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
    desc="Normalising images",
    show_progress=True,
    log_level="WARNING",
):
    """Load and process multiple images in parallel.

    Args:
        images (list): image or list of images(H,W) or (H,W,C) to normalise
        output_dtype (type, optional): Data type for output images. Defaults to np.uint8.
        normalisation_method (NormalisationMethod, optional): Normalisation method to use.
                                                Defaults to NormalisationMethod.CONVERSION_ONLY.
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

    Returns:
        list: List of images for successfully normalised images
    """
    # check if input is a single image array or a list of images
    if isinstance(images, np.ndarray) and (images.ndim == 2 or images.ndim == 3):
        # Single image array
        return_single = True
        n_output_channels = images.shape[-1] if images.ndim == 3 else 1
        images = [images]
    elif isinstance(images, list):
        if len(images) == 0:
            return []
        # List of images
        return_single = False
        n_output_channels = images[0].shape[-1] if images[0].ndim == 3 else 1

    elif isinstance(images, np.ndarray) and images.ndim == 4:
        # provide support if user provises an array instead of a list
        return_single = False
        n_output_channels = images.shape[-1]
    else:
        raise ValueError(
            f"Unsupported image format: {type(images)}, should be a list or a 2D, 3D array (single images) or a 4D array"
        )

    cfg = create_config(
        output_dtype=output_dtype,
        normalisation_method=normalisation_method,
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

    # Add a new logger configuration for console output
    logger.set_log_level(cfg.log_level)

    logger.debug(f"Setting LogLevel to {cfg.log_level.upper()}")

    logger.debug(
        f"Normalising {len(images)} images in parallel with normalisation: {cfg.normalisation_method}"
    )

    def normalise_single_image(image):
        try:
            image = _normalise_image(
                image,
                cfg,
            )
            if image is None:
                logger.error("Failed to normalise image")
                raise ValueError("Image normalisation failed. Check the image content.")
            return image
        except Exception as e:
            logger.error(f"Error loading {image}: {str(e)}")
            raise e

    # Use ThreadPoolExecutor for parallel loading
    with ThreadPoolExecutor(max_workers=cfg.num_workers) as executor:
        if show_progress:
            results = list(
                tqdm(
                    executor.map(normalise_single_image, images),
                    desc=desc,
                    total=len(images),
                )
            )
        else:
            results = list(executor.map(normalise_single_image, images))

    logger.debug(f"Successfully loaded {len(results)} of {len(images)} images")
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
