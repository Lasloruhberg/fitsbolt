# fitsbolt - A Python package for image loading and processing
# Copyright (C) <2025>  <Ruhberg>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the MIT or GPL-3.0 License

import math

import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
import warnings

from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod
from fitsbolt.cfg.create_config import create_config
from fitsbolt.cfg.logger import logger

# Lazy imports for heavy dependencies
_astropy_viz = None
_skimage_util = None


def _get_astropy_viz():
    """Lazy import of astropy.visualization components."""
    global _astropy_viz
    if _astropy_viz is None:
        from astropy.visualization import (
            ImageNormalize,
            LinearStretch,
            ZScaleInterval,
        )

        _astropy_viz = {
            "ImageNormalize": ImageNormalize,
            "LinearStretch": LinearStretch,
            "ZScaleInterval": ZScaleInterval,
        }
    return _astropy_viz


def _get_skimage_util():
    """Lazy import of skimage.util components."""
    global _skimage_util
    if _skimage_util is None:
        from skimage.util import img_as_ubyte, img_as_uint, img_as_float32

        _skimage_util = {
            "img_as_ubyte": img_as_ubyte,
            "img_as_uint": img_as_uint,
            "img_as_float32": img_as_float32,
        }
    return _skimage_util


def _type_conversion(data: np.ndarray, cfg) -> np.ndarray:
    """Convert the image data to the specified output dtype."""
    skimage_util = _get_skimage_util()
    if cfg.output_dtype == np.uint8:
        return skimage_util["img_as_ubyte"](data)
    elif cfg.output_dtype == np.uint16:
        return skimage_util["img_as_uint"](data)
    elif cfg.output_dtype == np.float32:
        return skimage_util["img_as_float32"](data)
    else:
        # Default to uint8 if output_dtype is not specified or not supported
        warnings.warn(f"Unsupported output dtype: {cfg.output_dtype}, defaulting to uint8")
        return skimage_util["img_as_ubyte"](data)


def _flatten_and_subsample(channel_data, n_samples) -> np.ndarray:
    """Flatten the data and subsample it to n_samples if n_samples is not None and smaller than the data size.
    When ``n_samples`` is set and smaller than the channel's pixel count, the
    percentile bounds (vmin/vmax) are estimated from a deterministic strided
    subsample rather than from every pixel.

    Returns: A 1D array of samples for computation, either the full flattened data or a strided subsample.

    """
    if n_samples is not None and channel_data.size > n_samples:
        flat = channel_data.reshape(-1)
        # A constant stride over the C-order-flattened image aliases against the
        # row length: when ``gcd(step, row_length) > 1`` the samples land on only
        # a few columns, so vmin/vmax get estimated from a vertical sliver of the
        # frame and miss spatially localised bright sources. Nudging ``step`` to
        # be coprime with the row length makes the stride walk every column while
        # keeping the sample count at ~``n_samples`` (so the speed-up is intact).
        row_length = channel_data.shape[-1]
        step = max(1, flat.size // n_samples)
        tries = 0
        while row_length > 1 and math.gcd(step, row_length) != 1 and tries < row_length:
            step += 1
            tries += 1
        sample = flat[::step]
        return sample
    else:
        return channel_data.reshape(-1)


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


def _compute_max_value(data, cfg):
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
        max_value = np.nanmax(
            _flatten_and_subsample(img_centre_region, cfg.normalisation.minmax_n_samples)
        )

    else:
        # Compute the maximum value of the image
        max_value = (
            cfg.normalisation.maximum_value
            if cfg.normalisation.maximum_value is not None
            else np.nanmax(_flatten_and_subsample(data, cfg.normalisation.minmax_n_samples))
        )

    return max_value


def _compute_min_value(data, cfg):
    """Compute the minimum value of the image for normalisation
    Args:
        data (numpy array): Input image array, can be high dynamic range
        cfg (DotMap or None): Configuration with optional normalisation values.
    Returns:
        float: Minimum value for normalisation
    """
    min_value = (
        cfg.normalisation.minimum_value
        if cfg.normalisation.minimum_value is not None
        else np.nanmin(_flatten_and_subsample(data, cfg.normalisation.minmax_n_samples))
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
    if not minimum < maximum:
        # would result in a black image
        minimum = np.nanmin(data)
        maximum = np.nanmax(data)
        if minimum < maximum:
            pass
        else:
            warnings.warn("Image maximum is not larger than minimum, using conversion only")
            return _conversiononly_normalisation(data, cfg=cfg)

    # scale to 0,1
    # ensure data is float32 or float64
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32)
    try:
        np.clip(data, minimum, maximum, out=data)
    except:  # noqa: E722
        # likely the user specified clips that do not match the dtype - this copies the array and casts
        data = np.clip(data, minimum, maximum)
    np.subtract(data, minimum, out=data)
    np.true_divide(data, maximum - minimum, out=data)

    # apply log stretch as in astropy
    a = cfg.normalisation.log_scale_a
    np.multiply(data, a, out=data)
    np.add(data, 1.0, out=data)
    np.log(data, out=data)
    np.true_divide(data, np.log(a + 1), out=data)
    # Convert back to uint8 range
    np.clip(data, 0, 1, out=data)
    return _type_conversion(data, cfg)


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
        # ensure data is float32 or float64
        if not np.issubdtype(data.dtype, np.floating):
            data = data.astype(np.float32)
        try:
            np.clip(data, minimum, maximum, out=data)
        except:  # noqa: E722
            # likely the user specified clips that do not match the dtype - this copies the array and casts
            data = np.clip(data, minimum, maximum)
        np.subtract(data, minimum, out=data)
        np.true_divide(data, maximum - minimum, out=data)
    else:
        warnings.warn(
            "Image maximum is not larger than minimum, only doing conversion normalisation"
        )
        return _conversiononly_normalisation(data, cfg)

    # Convert back to type range
    np.clip(data, 0, 1, out=data)
    return _type_conversion(data, cfg)


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

    viz = _get_astropy_viz()
    # Min Max value do not apply, also no constrain to center
    norm = viz["ImageNormalize"](
        data,
        interval=viz["ZScaleInterval"](
            n_samples=cfg.normalisation.zscale.n_samples,
            contrast=cfg.normalisation.zscale.contrast,
            max_reject=cfg.normalisation.zscale.max_reject,
            min_npixels=cfg.normalisation.zscale.min_npixels,
            krej=cfg.normalisation.zscale.krej,
            max_iterations=cfg.normalisation.zscale.max_iterations,
        ),
        stretch=viz["LinearStretch"](),
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
    try:
        np.clip(data, minimum, maximum, out=data)
    except:  # noqa: E722
        # likely the user specified clips that do not match the dtype - this copies the array and casts
        data = np.clip(data, minimum, maximum)

    if data.dtype == cfg.output_dtype:
        if np.issubdtype(cfg.output_dtype, np.floating):
            # For float output, ensure data is in [0,1] range later on
            # This is conversion only, so do not change the data
            return data

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
            # scale to 0,1
            np.subtract(data, minimum, out=data)
            # need result in a new copy as divide might induce dtype change
            data = np.true_divide(data, maximum - minimum)

        else:
            np.subtract(data, minimum, out=data)  # should return 0
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
        # ensure data is floating, normalise to 0,1 and clip
        if not np.issubdtype(data.dtype, np.floating):
            data = data.astype(np.float32)
        np.subtract(data, minimum, out=data)
        np.true_divide(data, maximum - minimum, out=data)
        np.clip(data, 0, 1, out=data)
        return _type_conversion(data, cfg)
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


def _percentile_clip_vmin_vmax(channel_data, clip_percentile, n_samples):
    """Obtain percentile values from a sample of points based on a symmetric clip interval

    When ``n_samples`` is set and smaller than the channel's pixel count, the
    percentile bounds (vmin/vmax) are estimated from a deterministic strided
    subsample rather than from every pixel. The percentile computation dominates
    the asinh stretch cost, so this is a large speed-up for a small bias in the
    bright tail. The stride is deterministic (not random, unlike astropy's
    ``PercentileInterval(n_samples=...)``) so repeated runs stay reproducible,
    which matters when the output feeds a downstream model. The stride is made
    coprime with the row length so the subsample walks every column rather than
    aliasing onto a few. Only vmin/vmax are affected; the ``AsinhStretch`` and
    clipping are identical to the exact path.

    Args:
        channel_data (np.ndarray): A single channel of image data.
        clip_percentile (float): Percentile width passed to PercentileInterval.
        n_samples (int or None): Subsample size, or None to use all pixels.

    Returns:
        tuple: The lower and upper percentile values.
    """
    sample = _flatten_and_subsample(channel_data.copy(), n_samples)
    lower = (100.0 - clip_percentile) / 2.0

    k1 = int((lower / 100) * (sample.size - 1))
    k2 = int(((100.0 - lower) / 100) * (sample.size - 1))
    sample.partition((k1, k2))
    return sample[k1], sample[k2]


def _apply_asinh_norm(data, vmin, vmax, scale, cfg, recompute=False):
    """Apply asinh normalisation to the data using the provided configuration.
    First clip and limit to 0,1, then apply astropy like asinh
    Returns:
        np.ndarray: The transformed image data in [0,1]"""
    denominator = vmax - vmin
    if denominator == 0:
        return np.zeros_like(data, dtype=np.float32)
    try:
        np.clip(data, vmin, vmax, out=data)
    except:  # noqa: E722
        data = np.clip(data, vmin, vmax)
    np.subtract(data, vmin, out=data)
    np.true_divide(data, (denominator), out=data)

    np.true_divide(data, scale, out=data)
    np.arcsinh(data, out=data)

    denominator = cfg.normalisation.get("precomputed_asinh_inverse_asinh_scale", None)
    if denominator is None or recompute:
        denominator = np.arcsinh(1.0 / scale)
    np.true_divide(data, denominator, out=data)
    return data


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
    # we want to keep it coloursafe if scale and clip both are lists of len 1
    colour_safe = False
    if isinstance(cfg.normalisation.asinh_scale, (list, tuple)) and isinstance(
        cfg.normalisation.asinh_clip, (list, tuple)
    ):
        if len(cfg.normalisation.asinh_scale) == 1 and len(cfg.normalisation.asinh_clip) == 1:
            colour_safe = True
            # precompute for speedup
            cfg.normalisation.precomputed_asinh_inverse_asinh_scale = np.arcsinh(
                1.0 / cfg.normalisation.asinh_scale[0]
            )

    scale = _expand(cfg.normalisation.asinh_scale, channels)
    clip = _expand(cfg.normalisation.asinh_clip, channels)

    # Get initial min and max and clip values if manual are set
    max_value = _compute_max_value(data, cfg)
    min_value = _compute_min_value(data, cfg)
    try:
        np.clip(data, min_value, max_value, out=data)
    except:  # noqa: E722
        # likely the user specified clips that do not match the dtype - this copies the array and casts
        data = np.clip(data, min_value, max_value)
    # ensure data is float32 or float64
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32)

    # Apply asinh normalisation & percentile clipping, potentially per-channel
    if channels == 1:
        vmin, vmax = _percentile_clip_vmin_vmax(
            data, clip[0], cfg.normalisation.percentile_n_samples
        )
        normalised = _apply_asinh_norm(data, vmin, vmax, scale[0], cfg)
    # Multi channel case: either colour-safe or per-channel normalisation
    else:
        if colour_safe:

            # compute all vmins and vmaxs first and then take max/min over all
            vmins = np.empty(channels)
            vmaxs = np.empty(channels)

            for channel_idx in range(channels):
                vmins[channel_idx], vmaxs[channel_idx] = _percentile_clip_vmin_vmax(
                    data[..., channel_idx],
                    clip[channel_idx],
                    cfg.normalisation.percentile_n_samples,
                )

            vmin = vmins.min()
            vmax = vmaxs.max()

            normalised = _apply_asinh_norm(data, vmin, vmax, scale[0], cfg)

        else:
            # normalise each channel individually
            normalised = np.zeros_like(data, dtype=np.float32)

            for channel_idx in range(channels):
                vmin, vmax = _percentile_clip_vmin_vmax(
                    data[..., channel_idx],
                    clip[channel_idx],
                    cfg.normalisation.percentile_n_samples,
                )
                normalised[..., channel_idx] = _apply_asinh_norm(
                    data[..., channel_idx], vmin, vmax, scale[channel_idx], cfg, recompute=True
                )

    # correct to 0-1 range and convert to uint8
    # check that the image is not entirely black
    first = normalised.flat[0]
    if np.any(normalised != first):
        return _type_conversion(np.clip(normalised, 0.0, 1.0), cfg)

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


def _find_mean_of_normalised(normalised_data, cfg, channel_index):
    """Find the midtones balance parameter m for the given normalised data.

    Args:
        normalised_data is expected to be in the range [0, 1] and m is computed based on the mean of the data.
        cfg is a fitsbolt configuration object that contains the desired mean for the midtones normalisation.
        channel_index is the index of the channel for which to compute the midtones balance parameter.

    Returns:
        float: The midtones balance parameter m.
    """
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

    # failsafe to avoid crashes when channel is longer than expected
    if channel_index >= len(cfg.normalisation.midtones.desired_mean):
        channel_index = -1
    alpha = cfg.normalisation.midtones.desired_mean[channel_index]
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
    if min_value >= max_value:
        warnings.warn("Image maximum is not larger than minimum, returning conversion only.")
        return _conversiononly_normalisation(data, cfg=cfg)
    try:
        np.clip(data, min_value, max_value, out=data)
    except:  # noqa: E722
        # likely the user specified clips that do not match the dtype - this copies the array and casts
        data = np.clip(data, min_value, max_value)

    # ensure data is float32 or float64
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32)

    data_is_2d = False
    if data.ndim == 2:
        # create dummy channel index
        data_is_2d = True
        data = np.expand_dims(data, axis=-1)

    # if there is only one percentile and 1 desired mean - use coulr safe mode
    # if the user specifies more than one percentile or more than one desired mean, channels are normalised individually
    colour_safe = (
        len(cfg.normalisation.midtones.percentile) == 1
        and len(cfg.normalisation.midtones.desired_mean) == 1
    )

    if colour_safe:
        # create a for loop over the channel to calculate m and apply MTF on a channel basis
        max_values = np.empty(data.shape[-1])
        min_values = np.empty(data.shape[-1])

        for channel_idx in range(data.shape[-1]):
            # do a channel-wise percentile clip

            if cfg.normalisation.midtones.percentile:
                min_value, max_value = _percentile_clip_vmin_vmax(
                    data[..., channel_idx],
                    cfg.normalisation.midtones.percentile[0],
                    cfg.normalisation.percentile_n_samples,
                )
            else:
                # Find the appropriate midtones balance parameter m
                max_value = _compute_max_value(data[..., channel_idx], cfg)
                min_value = _compute_min_value(data[..., channel_idx], cfg)
            max_values[channel_idx] = max_value
            min_values[channel_idx] = min_value
        # include necessary clipping
        min_value = np.min(min_values)
        max_value = np.max(max_values)
        try:
            np.clip(data, min_value, max_value, out=data)
        except:  # noqa: E722
            # likely the user specified clips that do not match the dtype - this copies the array and casts
            data = np.clip(data, min_value, max_value)
        # Skip MTF for constant channels (avoids division by zero)
        if min_value >= max_value:
            data[...] = 0.0

        else:
            np.subtract(data, min_value, out=data)
            np.true_divide(data, max_value - min_value, out=data)
            m = _find_mean_of_normalised(data, cfg, channel_index=0)

            # Apply the MTF to the image
            data = _apply_midtones_on_normalised_data(data, m)

    # if user wants the non-colousafe mode
    else:
        # create a for loop over the channel to calculate m and apply MTF on a channel basis
        for channel_idx in range(data.shape[-1]):
            # do a channel-wise percentile clip
            if cfg.normalisation.midtones.percentile:
                if channel_idx >= len(cfg.normalisation.midtones.percentile):
                    percentile = cfg.normalisation.midtones.percentile[-1]
                else:
                    percentile = cfg.normalisation.midtones.percentile[channel_idx]
                min_value, max_value = _percentile_clip_vmin_vmax(
                    data[..., channel_idx],
                    percentile,
                    cfg.normalisation.percentile_n_samples,
                )
            else:
                # Find the appropriate midtones balance parameter m
                max_value = _compute_max_value(data[..., channel_idx], cfg)
                min_value = _compute_min_value(data[..., channel_idx], cfg)
            # include necessary clipping
            data[..., channel_idx] = np.clip(data[..., channel_idx], min_value, max_value)

            # Skip MTF for constant channels (avoids division by zero)
            if min_value >= max_value:
                data[..., channel_idx] = 0.0
                continue

            np.subtract(data[..., channel_idx], min_value, out=data[..., channel_idx])
            np.true_divide(
                data[..., channel_idx], max_value - min_value, out=data[..., channel_idx]
            )

            m = _find_mean_of_normalised(data[..., channel_idx], cfg, channel_index=channel_idx)
            # Apply the MTF to the image
            transformed_channel = _apply_midtones_on_normalised_data(data[..., channel_idx], m)

            data[..., channel_idx] = transformed_channel
    if data_is_2d:
        data = np.squeeze(data, axis=-1)
    # scale entire image to 0,1 and do type conversion
    max_value = _compute_max_value(data, cfg)
    min_value = _compute_min_value(data, cfg)
    if min_value < max_value:
        try:
            np.clip(data, min_value, max_value, out=data)
        except:  # noqa: E722
            # likely the user specified clips that do not match the dtype - this copies the array and casts
            data = np.clip(data, min_value, max_value)
        np.subtract(data, min_value, out=data)
        np.true_divide(data, max_value - min_value, out=data)
        np.clip(data, 0, 1, out=data)
        return _type_conversion(data, cfg)
    else:
        warnings.warn("Image maximum is not larger than minimum, returning zeros.")

        return np.zeros_like(data, dtype=cfg.output_dtype)


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


def _worker_normalise_image(image, cfg):
    """Module-level worker for ProcessPoolExecutor compatibility."""
    try:
        image = _normalise_image(image, cfg)
        if image is None:
            raise ValueError("Image normalisation failed. Check the image content.")
        return image
    except Exception as e:
        logger.error(f"Error normalising image: {str(e)}")
        raise e


def normalise_images(
    images,
    output_dtype=np.uint8,
    normalisation_method=NormalisationMethod.CONVERSION_ONLY,
    num_workers=4,
    norm_maximum_value=None,
    norm_minimum_value=None,
    norm_crop_for_maximum_value=None,
    norm_minmax_samples=None,
    norm_percentile_samples=None,
    norm_log_calculate_minimum_value=False,
    norm_log_scale_a=1000.0,
    norm_asinh_scale=[0.7],
    norm_asinh_clip=[99.8],
    norm_asinh_n_samples=None,
    norm_zscale_n_samples=1000,
    norm_zscale_contrast=0.25,
    norm_zscale_max_reject=0.5,
    norm_zscale_min_pixels=5,
    norm_zscale_krej=2.5,
    norm_zscale_max_iter=5,
    norm_midtones_percentile=[99.8],
    norm_midtones_desired_mean=[0.2],
    norm_midtones_crop=None,
    desc="Normalising images",
    show_progress=True,
    log_level="WARNING",
    use_multiprocessing=False,
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
        norm_minmax_samples (int, optional): If set, the min/max bounds (vmin/vmax) are estimated
                                            from a deterministic strided subsample of this many pixels per
                                            channel instead of all pixels. Trades a small bias in the bright
                                            tail for a large reduction in cost. Defaults to None (use all pixels, exact).
        norm_percentile_samples (int, optional): If set, the percentile bounds (vmin/vmax) are estimated
                                                from a deterministic strided subsample of this many pixels per
                                                channel instead of all pixels. Trades a small bias in the bright
                                                tail for a large reduction in cost. Defaults to None (use all pixels, exact).

        Default Log settings
            norm_log_calculate_minimum_value (bool, optional): If True, calculates the minimum value for log scaling.
                                Defaults to False.
            norm_log_scale_a (float, optional): Scale factor for astropy log_stretch. Defaults to 1000.0.
        Default Asinh settings
            norm_asinh_scale (list, optional): Scale factors for asinh normalisation,
                                                should have the length of n_output_channels or 1. Defaults to [0.7].
            norm_asinh_clip (list, optional): Clip values for asinh normalisation,
                                                should have the length of n_output_channels or 1. Defaults to [99.8].
            norm_asinh_n_samples (int, optional): If set, the asinh percentile bounds are estimated from a
                                                deterministic strided subsample of this many pixels per channel
                                                rather than all pixels, trading a small bright-tail bias for speed.
                                                Defaults to None (exact).
        Default ZScale settings (from astropy ZScaleInterval):
            norm_zscale_n_samples (int, optional): Number of samples for zscale normalisation. Defaults to 1000.
            norm_zscale_contrast (float, optional): Contrast for zscale normalisation. Defaults to 0.25.
            norm_zscale_max_reject (float, optional): Maximum rejection fraction for zscale normalisation. Defaults to 0.5.
            norm_zscale_min_pixels (int, optional): Minimum number of pixels that must remain after rejection
                                                    for zscale normalisation. Defaults to 5.
            norm_zscale_krej (float, optional): The number of sigma used for the rejection. Defaults to 2.5.
            norm_zscale_max_iter (int, optional): Maximum number of iterations for zscale normalisation. Defaults to 5.

        Default MTF settings:
            norm_midtones_percentile (list(float), optional): Percentile for MTF applied to each channel, in ]0., 100.].
                                                        Length one for colour-safe or lenght n_channels for per-channel MTF.
                                                        Defaults to [99.8].
            norm_midtones_desired_mean (list(float), optional): Desired mean for MTF, in [0, 1]. Defaults to [0.2].
                                                        Length one for colour-safe or lenght n_channels for per-channel MTF.
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
        norm_minmax_samples=norm_minmax_samples,
        norm_percentile_samples=norm_percentile_samples,
        norm_asinh_scale=norm_asinh_scale,
        norm_asinh_clip=norm_asinh_clip,
        norm_asinh_n_samples=norm_asinh_n_samples,
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

    worker_fn = partial(_worker_normalise_image, cfg=cfg)
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
