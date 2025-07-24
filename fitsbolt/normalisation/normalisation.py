import numpy as np
from loguru import logger


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
        logger.warning(f"Unsupported output dtype: {cfg.output_dtype}, defaulting to uint8")
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
        logger.warning("Crop size is larger than image size, returning original image")
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
        max_value = np.max(img_centre_region)

    else:
        # Compute the maximum value of the image
        max_value = (
            cfg.normalisation.maximum_value
            if cfg.normalisation.maximum_value is not None
            else np.max(data)
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
        else np.min(data)
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
        norm = ImageNormalize(data, vmin=minimum, vmax=maximum, stretch=LogStretch(), clip=True)
    else:
        logger.warning(
            "Image minimum value is larger than maximum, ignoring boundaries and using a LinearInterval"
        )
        norm = ImageNormalize(data, vmin=None, vmax=None, stretch=LogStretch(), clip=True)
    img_normalised = norm(data)  # range 0,1
    # Convert back to uint8 range
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
        logger.warning("Zscale normalisation: constant image detected, using fallback conversion.")
        return _conversiononly_normalisation(data, cfg)

    # Min Max value do not apply, also no constrain to center
    norm = ImageNormalize(data, interval=ZScaleInterval(), stretch=LinearStretch(), clip=True)
    img_normalised = norm(data)  # range 0,1
    if np.max(img_normalised) > np.min(img_normalised):
        # Convert back to specified dtype
        return _type_conversion(img_normalised, cfg)
    else:
        logger.warning(
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
        numpy array: A converted image in the specified output dtype
    """
    # If input dtype already matches the requested output dtype, return as is
    if data.dtype == cfg.output_dtype:
        return data

    # Handle specific direct conversions for better precision
    if cfg.output_dtype == np.uint8:
        if data.dtype == np.uint16:
            # Direct conversion from uint16 to uint8 with proper scaling
            return _type_conversion(data / 65535.0, cfg)  # 65535 = 2^16 - 1

        elif data.dtype == np.float32 or data.dtype == np.float64:
            # For floating point data that's already in [0,1] range
            if 0.0 <= np.min(data) <= np.max(data) <= 1.0:
                return _type_conversion(data, cfg)

    elif cfg.output_dtype == np.uint16:
        if data.dtype == np.uint8:
            # Direct conversion from uint8 to uint16 with proper scaling
            return _type_conversion(data / 255.0, cfg)  # Scale to [0,1] then convert

        elif data.dtype == np.float32 or data.dtype == np.float64:
            # For floating point data that's already in [0,1] range
            if 0.0 <= np.min(data) <= np.max(data) <= 1.0:
                return _type_conversion(data, cfg)

    elif cfg.output_dtype == np.float32:
        if data.dtype == np.uint8:
            # Convert uint8 directly to float32 [0,1] range
            return _type_conversion(data / 255.0, cfg)

        elif data.dtype == np.uint16:
            # Convert uint16 directly to float32 [0,1] range
            return _type_conversion(data / 65535.0, cfg)

    # For any other case, use normalized conversion
    # get min or max from config if available
    maximum = _compute_max_value(data, cfg)
    minimum = _compute_min_value(data, cfg)

    # ensure valid range
    if maximum > minimum:
        norm = ImageNormalize(data, vmin=minimum, vmax=maximum, clip=True)
        img_normalised = norm(data)  # range 0,1
        return _type_conversion(img_normalised, cfg)
    else:
        logger.warning(
            "Conversion normalisation: Image minimum value is larger than maximum, setting image to 0"
        )
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
                f"Parameter asinh_scale or asinh_clip: {value!r} has length {arr.size}, expected {length}."
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
        or a three-element sequence.
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
        logger.warning(
            "Image maximum value is not larger than minimum, using minimal normalisation instead. Check settings"
        )
        return _conversiononly_normalisation(data, cfg=cfg)


def normalise_image(data, cfg):
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
    elif method == NormalisationMethod.CONVERSION_ONLY:
        return _conversiononly_normalisation(data, cfg=cfg)
    elif method == NormalisationMethod.ZSCALE:
        return _zscale_normalisation(data, cfg=cfg)
    elif method == NormalisationMethod.ASINH:
        return _asinh_normalisation(data, cfg=cfg)
    else:
        logger.critical(f"Normalisation method {method} not implemented")
        return _conversiononly_normalisation(data, cfg=cfg)
