# fitsbolt - A Python package for image loading and processing
# Copyright (C) <2025>  <Ruhberg>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the MIT License

import numbers
import numpy as np
import warnings
from dotmap import DotMap

from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod
from fitsbolt.cfg.logger import logger

SUPPORTED_IMAGE_EXTENSIONS = {".fits", ".jpg", ".jpeg", ".png", ".tiff", ".tif"}


def create_config(
    output_dtype=np.uint8,
    size=[224, 224],
    fits_extension=None,
    interpolation_order=1,
    n_output_channels=3,
    normalisation_method=NormalisationMethod.CONVERSION_ONLY,
    channel_combination=None,
    num_workers=4,
    norm_maximum_value=None,
    norm_minimum_value=None,
    norm_log_calculate_minimum_value=False,
    norm_log_scale_a=1000.0,
    norm_crop_for_maximum_value=None,
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
    log_level="WARNING",
    force_dtype=True,
):
    """Create a configuration object for loading and processing astronomical data.

    Args:
        output_dtype (type, optional): Data type for output images. Defaults to np.uint8.
        size (list, optional): Target size for image resizing. Defaults to [224, 224]. If None, no resizing.
        fits_extension (list, optional): Extension(s) to use when loading FITS files. Defaults to None.
        interpolation_order (int, optional): Order of interpolation for resizing with skimage, 0-5. Defaults to 1.
        n_output_channels (int,optional): number of output channels. Defaults to 3.
        normalisation_method (NormalisationMethod, optional): Method for normalising images.
                            Defaults to NormalisationMethod.CONVERSION_ONLY.
        channel_combination (np.ndarray, optional): n_output x fits_extension sized np array for channel mapping& lerp.
                            Defaults to None, which will map extensions to channels either
                            1:1 if applicable or 1:n_output if only 1 input is provided.
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
            norm_zscale_min_pixels (int, optional): Minimum number of pixels that must remain after rejection for
                                                    zscale normalisation. Defaults to 5.
            norm_zscale_krej (float, optional): The number of sigma used for the rejection. Defaults to 2.5.
            norm_zscale_max_iter (int, optional): Maximum number of iterations for zscale normalisation. Defaults to 5.

        Default MTF settings:
            norm_midtones_percentile (float, optional): Percentile for MTF applied to each channel, in ]0., 100.].
                                                        Defaults to 99.8.
            norm_midtones_desired_mean (float, optional): Desired mean for MTF, in [0, 1]. Defaults to 0.2.
            norm_midtones_crop (tuple, optional): Crops the image to a size of (h,w) around the center to determine the mean in
                                                    Defaults to None.

        log_level (str, optional): Logging level. Defaults to "SUCCESS".
        force_dtype (bool, optional): If True, forces the output to maintain the original dtype after tensor operations
                            like channel combination. Defaults to True.


    Returns:
        _type_: _description_
    """
    cfg = DotMap(_dynamic=False)
    # Settings
    cfg.log_level = log_level
    cfg.output_dtype = output_dtype
    cfg.size = size  # tuple of (height, width)
    cfg.n_output_channels = n_output_channels  # int, normally 3 for R,G,B
    cfg.num_workers = num_workers
    cfg.force_dtype = force_dtype  # Force output to maintain original dtype after tensor operations
    # order of interpolation for resizing with skimage, 0-5
    cfg.interpolation_order = interpolation_order
    # Normalisation settings
    cfg.normalisation_method = normalisation_method
    # Optional normalisation settings
    cfg.normalisation = DotMap()
    cfg.normalisation.maximum_value = norm_maximum_value  # None or float
    cfg.normalisation.minimum_value = norm_minimum_value  # None or float
    cfg.normalisation.crop_for_maximum_value = (
        norm_crop_for_maximum_value  # None or integer tuple (height, width)
    )
    # Bool, if False assumes min value to be 0 or cfg.normalisation.minimum_value if not None
    cfg.normalisation.log_calculate_minimum_value = norm_log_calculate_minimum_value
    cfg.normalisation.log_scale_a = norm_log_scale_a  # float, scale factor for astropy log_stretch
    # only used if cfg.normalisation_method == NormalisationMethod.ASINH:
    # asinh_scale list of 3 floats > 0, defining the scale for each channel (lower = higher stretch):
    cfg.normalisation.asinh_scale = norm_asinh_scale
    # asinh_clip list of 3 floats in ]0.,100.], defining the clip for each channel:
    cfg.normalisation.asinh_clip = norm_asinh_clip

    # ZSCALE settings
    cfg.normalisation.zscale = DotMap()
    cfg.normalisation.zscale.n_samples = norm_zscale_n_samples  # int, number of samples for zscale
    cfg.normalisation.zscale.contrast = norm_zscale_contrast  # float, contrast for zscale
    # float, maximum rejection fraction for zscale:
    cfg.normalisation.zscale.max_reject = norm_zscale_max_reject
    # int, minimum number of pixels for zscale:
    cfg.normalisation.zscale.min_npixels = norm_zscale_min_pixels
    cfg.normalisation.zscale.krej = norm_zscale_krej  # float, number of sigma for zscale
    # int, maximum number of iterations for zscale:
    cfg.normalisation.zscale.max_iterations = norm_zscale_max_iter

    # MTF settings
    cfg.normalisation.midtones = DotMap()
    # float, in ]0., 100.] : percentile for MTF applied to each channel
    cfg.normalisation.midtones.percentile = norm_midtones_percentile
    # float in [0,1], desired mean for MTF
    cfg.normalisation.midtones.desired_mean = norm_midtones_desired_mean
    cfg.normalisation.midtones.crop = norm_midtones_crop

    # FITS file handling settings
    # Extension(s) to use when loading FITS files (can be int, string, or list of int/string)
    cfg.fits_extension = fits_extension
    # Dictionary defining how to combine FITS extensions into RGB channels, should contain lists of the same length
    # as cfg.fits_extension, or empty lists - then the first three extensions will be used for RGB
    if fits_extension is not None:
        # Convert single extension to list format
        if not isinstance(fits_extension, (list, tuple, np.ndarray)):
            fits_extension = [fits_extension]  # Convert to list for consistent handling
            cfg.fits_extension = fits_extension

        if channel_combination is None:
            if not (len(fits_extension) in [1, n_output_channels]):
                # fits extension does not match 1 channel (castable to greyscale) or n_output

                raise ValueError(
                    "Length of fits_extensions does not match the specified number of output channels and"
                    + "no mapping via channelcombination is provided."
                    + f"Length fits_extension: {np.array(fits_extension).size}"
                    + f"Specified output channels: {n_output_channels}"
                    + f"-> set channel_combination to be a {n_output_channels}x{np.array(fits_extension).size}"
                )
            else:
                # selecting n fits extensions but no combination will lead to a 1-1 mapping
                extension_size = len(fits_extension)
                combination_array = np.zeros((n_output_channels, extension_size))
                if combination_array.shape[0] == combination_array.shape[1]:
                    # Identity matrix mapping
                    for i in range(0, combination_array.shape[0]):
                        combination_array[i, i] = 1
                else:
                    # For the case where fits_extension has only 1 element
                    if extension_size == 1:
                        # Map all output channels to the single input channel
                        for i in range(0, combination_array.shape[0]):
                            combination_array[i, 0] = 1
                cfg.channel_combination = combination_array
        else:

            cfg.channel_combination = (
                channel_combination  # n_output x fits_extension sizes np array
            )
    validate_config(cfg)
    return cfg


def _return_required_and_optional_keys():
    """
    Returns the configuration parameters in a unified format.

    Returns:
        dict: Dictionary with parameter_name as key and [dtype, min, max, optional, allowed_values] as value
              - dtype: expected data type (str, int, float, bool, list, tuple, 'directory', 'file', 'special')
              - min: minimum value (None if not applicable)
              - max: maximum value (None if not applicable)
              - optional: True if parameter is optional, False if required
              - allowed_values: list of allowed values (None if not applicable)
    """
    config_spec = {
        # Required positive integers
        "num_workers": [int, 1, None, False, None],
        "n_output_channels": [int, 1, None, False, None],
        # Required boolean parameters
        "normalisation.log_calculate_minimum_value": [bool, None, None, False, None],
        "force_dtype": [bool, None, None, False, None],
        # Required parameters with allowed values
        "log_level": [
            str,
            None,
            None,
            False,
            [
                "TRACE",
                "DEBUG",
                "INFO",
                "WARNING",
                "ERROR",
                "CRITICAL",
            ],
        ],
        # Required special parameters
        "size": ["special_size", None, None, False, None],
        "normalisation_method": ["special_normalisation_method", None, None, False, None],
        "normalisation": ["special_DotMap", None, None, False, None],
        "normalisation.asinh_scale": ["special_asinh_scale", None, None, False, None],
        "normalisation.asinh_clip": ["special_asinh_clip", None, None, False, None],
        "interpolation_order": [int, 0, 5, False, None],  # 0-5 for skimage interpolation"
        "output_dtype": [type, None, None, False, None],
        # Optional numeric parameters
        "normalisation.maximum_value": [float, None, None, True, None],
        "normalisation.minimum_value": [float, None, None, True, None],
        "normalisation.log_scale_a": [float, 0.0, None, False, None],
        # Optional special parameters
        "normalisation.crop_for_maximum_value": ["special_crop", None, None, True, None],
        "fits_extension": ["special_fits_extension", None, None, True, None],
        "channel_combination": ["special_channel_combination", None, None, True, None],
        # further params
        "normalisation.zscale": ["special_DotMap", None, None, True, None],
        "normalisation.zscale.n_samples": [int, None, None, True, None],
        "normalisation.zscale.contrast": [float, 0.0, 1.0, True, None],
        "normalisation.zscale.max_reject": [float, 0.0, 1.0, True, None],
        "normalisation.zscale.min_npixels": [int, 1, None, True, None],
        "normalisation.zscale.krej": [float, 0.0001, None, True, None],
        "normalisation.zscale.max_iterations": [int, 1, 100, True, None],
        "normalisation.midtones": ["special_DotMap", None, None, True, None],
        "normalisation.midtones.percentile": [float, 0.0, 100.0, True, None],
        "normalisation.midtones.desired_mean": [float, 0.0, 1.0, True, None],
        "normalisation.midtones.crop": ["special_crop", None, None, True, None],
    }

    return config_spec


def _get_nested_value(cfg: DotMap, key: str):
    """Get a nested value from the config using dot notation.

    Args:
        cfg (DotMap): Configuration object
        key (str): Key in dot notation (e.g., 'normalisation.maximum_value')

    Returns:
        Any: Value from the config
    """
    current = cfg
    for part in key.split("."):
        try:
            current = current[part]
        except (KeyError, TypeError):
            raise ValueError(f"Missing key in config: {key}")
    return current


def _get_all_keys(cfg: DotMap, parent_key: str = ""):
    """Get keys of the configuration in dot notation.

    Args:
        cfg (DotMap): Configuration dotmap
        parent_key (str, optional): Parent key for nested values."".

    Returns:
        set: Set of all keys in dot notation
    """
    keys = set()
    for key, value in cfg.items():
        current_key = f"{parent_key}.{key}" if parent_key else key
        keys.add(current_key)
        if isinstance(value, DotMap):
            keys.update(_get_all_keys(value, current_key))
    return keys


def validate_config(cfg: DotMap, check_paths: bool = True) -> None:
    """Validate configuration against required and optional keys specification.

    Args:
        cfg (DotMap): Configuration to validate
        check_paths (bool): Whether to check if file and directory paths exist

    Raises:
        ValueError: If configuration is invalid
    """
    # Get configuration specification
    config_spec = _return_required_and_optional_keys()

    # Keep track of checked keys
    expected_keys = set()

    # Validate each parameter
    for param_name, (dtype, min_val, max_val, optional, allowed_values) in config_spec.items():
        expected_keys.add(param_name)

        # Try to get the value, handle missing optional parameters
        try:
            value = _get_nested_value(cfg, param_name)
        except ValueError:
            if optional:
                continue  # Skip missing optional parameters
            else:
                raise ValueError(
                    f"Missing required parameter: {param_name}"
                    + f"(type: {dtype.__name__ if hasattr(dtype, '__name__') else dtype})"
                )
        # get value n_output_channels
        n_output_channels = cfg.n_output_channels if "n_output_channels" in cfg else 3
        # get current log level
        current_log_level = cfg.log_level if "log_level" in cfg else "WARNING"

        # Skip validation for None values on optional parameters
        if value is None and optional:
            continue

        # Helper function to format constraint info
        def _format_constraints():
            constraints = []
            if min_val is not None:
                constraints.append(f"min: {min_val}")
            if max_val is not None:
                constraints.append(f"max: {max_val}")
            if allowed_values is not None:
                constraints.append(f"allowed: {allowed_values}")
            return f" ({', '.join(constraints)})" if constraints else ""

        # Validate based on data type
        if dtype == str:
            if not isinstance(value, str):
                raise ValueError(
                    f"{param_name} must be a string, got {type(value).__name__}{_format_constraints()}"
                )
            # Check allowed values for string types
            if allowed_values is not None and value not in allowed_values:
                raise ValueError(f"{param_name} must be one of {allowed_values}, got '{value}'")

        elif dtype == int:
            if not isinstance(value, int):
                raise ValueError(
                    f"{param_name} must be an integer, got {type(value).__name__}{_format_constraints()}"
                )
            if min_val is not None and value < min_val:
                raise ValueError(
                    f"{param_name} must be >= {min_val}, got {value}{_format_constraints()}"
                )
            if max_val is not None and value > max_val:
                raise ValueError(
                    f"{param_name} must be <= {max_val}, got {value}{_format_constraints()}"
                )
            if allowed_values is not None and value not in allowed_values:
                raise ValueError(f"{param_name} must be one of {allowed_values}, got {value}")

        elif dtype == float:
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"{param_name} must be a number, got {type(value).__name__}{_format_constraints()}"
                )
            if min_val is not None and value < min_val:
                raise ValueError(
                    f"{param_name} must be >= {min_val}, got {value}{_format_constraints()}"
                )
            if max_val is not None and value > max_val:
                raise ValueError(
                    f"{param_name} must be <= {max_val}, got {value}{_format_constraints()}"
                )
            if allowed_values is not None and value not in allowed_values:
                raise ValueError(f"{param_name} must be one of {allowed_values}, got {value}")

        elif dtype == bool:
            if not isinstance(value, bool):
                raise ValueError(f"{param_name} must be a boolean, got {type(value).__name__}")

        elif dtype == type:
            if not (isinstance(value, type) or isinstance(value, np.dtype)):
                raise ValueError(
                    f"{param_name} must be a (numpy) dtype, got {type(value).__name__}{_format_constraints()}"
                )
            if not issubclass(value, numbers.Number):
                raise ValueError(
                    f"{param_name} must be a (numpy) dtype, got {type(value).__name__}{_format_constraints()}"
                )
            # No min/max/allowed values for dtypes
        # Handle special validation cases
        elif dtype == "special_size":
            if value is not None:
                if not isinstance(value, (list, tuple)) or len(value) != 2:
                    raise ValueError(
                        f"{param_name} must be a list or tuple of length 2, got {type(value).__name__}"
                        + f"with length {len(value) if hasattr(value, '__len__') else 'unknown'}"
                    )

        elif dtype == "special_normalisation_method":
            if not isinstance(value, NormalisationMethod):
                raise ValueError(
                    f"{param_name} must be a NormalisationMethod enum value, got {type(value).__name__}"
                )

        elif dtype == "special_DotMap":
            if not isinstance(value, DotMap):
                raise ValueError(f"{param_name} must be a DotMap, got {type(value).__name__}")

        elif dtype == "special_asinh_scale":
            if not isinstance(value, (list, tuple, int, float)):
                raise ValueError(
                    f"{param_name} must be a number or list/tuple of {n_output_channels} or 1 numbers > 0,"
                    + f" got {type(value).__name__}"
                )
            if isinstance(value, (list, tuple)):
                if len(value) != n_output_channels and len(value) != 1:
                    raise ValueError(
                        f"{param_name} if list/tuple, must have length {n_output_channels} or 1, got length {len(value)}"
                    )
                if not all(isinstance(x, (int, float)) for x in value):
                    raise ValueError(
                        f"{param_name} values must be numbers, got types: {[type(x).__name__ for x in value]}"
                    )
                if not all(0 < x for x in value):
                    raise ValueError(f"{param_name} values must be > 0, got: {value}")
            else:
                # Single value
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"{param_name} must be a number > 0, got {type(value).__name__}"
                    )
                if not (0 < value):
                    raise ValueError(f"{param_name} must > 0, got: {value}")

        elif dtype == "special_asinh_clip":
            if not isinstance(value, (list, tuple, int, float)):
                raise ValueError(
                    f"{param_name} must be a number or list/tuple of n_output_channels numbers in ]0,100.],"
                    + f" got {type(value).__name__}"
                )
            if isinstance(value, (list, tuple)):
                if len(value) != n_output_channels and len(value) != 1:
                    raise ValueError(
                        f"{param_name} if list/tuple, must have length {n_output_channels} or 1, got length {len(value)}"
                    )
                if not all(isinstance(x, (int, float)) for x in value):
                    raise ValueError(
                        f"{param_name} values must be numbers, got types: {[type(x).__name__ for x in value]}"
                    )
                if not all(0 < x <= 100 for x in value):
                    raise ValueError(f"{param_name} values must be in range ]0,100.], got: {value}")
            else:
                # Single value
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"{param_name} must be a number in ]0,100.], got {type(value).__name__}"
                    )
                if not (0 < value <= 100):
                    raise ValueError(f"{param_name} must be in range ]0,100.], got: {value}")

        elif dtype == "special_crop":
            if value is not None:
                if not isinstance(value, (tuple, list)) or len(value) != 2:
                    raise ValueError(
                        f"{param_name} if set, must be a tuple of two integers, got {type(value).__name__}"
                    )
                if not all(isinstance(x, int) for x in value):
                    raise ValueError(
                        f"{param_name} values must be integers, got types: {[type(x).__name__ for x in value]}"
                    )

        elif dtype == "special_fits_extension":
            if value is not None:
                if isinstance(value, list):
                    # if no combination parameters are set
                    if len(value) != cfg.channel_combination.shape[1]:
                        raise ValueError(
                            f"{param_name} must be a str/int or list of strings/ints of length 1, n_output ="
                            + f"{cfg.n_output_channels}, or match channel_combination.shape[1] = "
                            + f"{cfg.channel_combination.shape[1]}, got list of length {len(value)}"
                        )

                    for v in value:
                        if not isinstance(v, (str, int)):
                            raise ValueError(
                                f"{param_name} list elements must be str or int, got {type(v).__name__}"
                            )
                elif not isinstance(value, (str, int)):
                    raise ValueError(
                        f"{param_name} must be a str/int or list of strings/ints, got {type(value).__name__}"
                    )
        elif dtype == "special_channel_combination":

            if not isinstance(value, np.ndarray):
                raise ValueError(f"{param_name} must be a numpyarray")
            if np.any(value < 0):
                raise ValueError(f"{param_name} must not have negative values")
            # TODO check 2dimensionality
            for i in range(0, value.shape[0]):
                if np.any(np.allclose(np.sum(value[i, :]), 0)):
                    raise ValueError(
                        f"{param_name} values for channel '{i}' must not sum to zero, got {value[i, :]}"
                    )
            if not value.shape[0] == cfg.n_output_channels and not value.shape[1] == len(
                cfg.fits_extension
            ):
                raise ValueError(
                    f"{param_name} channel mapping shape must reflect input and output shapes:"
                    + f" expected n output {cfg.n_output_channels} & n input {len(cfg.fits_extension)}"
                    + f" got {value.shape[0]}x {value.shape[1]}"
                )

        else:
            raise ValueError(f"Unknown data type for {param_name}: {dtype}")

    # Custom cross-parameter validation
    if "normalisation" in cfg:
        if (
            hasattr(cfg.normalisation, "maximum_value")
            and hasattr(cfg.normalisation, "minimum_value")
            and isinstance(cfg.normalisation.maximum_value, (int, float))
            and isinstance(cfg.normalisation.minimum_value, (int, float))
        ):
            if cfg.normalisation.maximum_value <= cfg.normalisation.minimum_value:
                raise ValueError(
                    f"normalisation.maximum_value {cfg.normalisation.maximum_value} must be larger than "
                    f"normalisation.minimum_value {cfg.normalisation.minimum_value}"
                )

    # Check for unexpected keys
    actual_keys = _get_all_keys(cfg)
    unexpected_keys = actual_keys - expected_keys

    if unexpected_keys:
        warnings.warn(f"Found unexpected keys in config: {sorted(unexpected_keys)}")
        if current_log_level in ["DEBUG", "INFO", "TRACE"]:
            logger.info("Config: validation partially successful")
    else:
        if current_log_level in ["DEBUG", "INFO", "TRACE"]:
            logger.info("Config: validation successful")


def recompute_config_channel_combination(cfg):
    """Recomputes the channel combination mapping based on the current configuration.

    Args:
        cfg (Config): The configuration object containing the current settings.
        Must include n_expected_channels, should only be called after _read_image

    Raises:
        ValueError: If the channel combination mapping cannot be determined.
    """
    # no channel combination, eg not fits extensions specified
    # get n_expected_channels from read, compare to n_output_channels and create a 1_to n mapping
    # if n_expected=1 or 1:1 raise a Value Error otherwise
    # cfg.n_expected_channels is stored as a list, so extract the actual number
    actual_expected_channels = (
        cfg.n_expected_channels[0]
        if isinstance(cfg.n_expected_channels, list)
        else cfg.n_expected_channels
    )

    if actual_expected_channels == 1:
        channel_combination = np.ones((cfg.n_output_channels, actual_expected_channels))
    elif actual_expected_channels == cfg.n_output_channels:
        channel_combination = np.eye(cfg.n_output_channels)
    elif actual_expected_channels == 4 and cfg.n_output_channels == 3:
        # Common case: RGBA to RGB, drop the alpha channel
        channel_combination = np.eye(3, 4)
    else:
        raise ValueError(
            f"From files got {actual_expected_channels} expected channels, "
            f"but requested {cfg.n_output_channels} output channels. "
            "Cannot automatically create a valid channel combination mapping. "
            "Please provide the channel_combination parameter"
        )

    # Store the computed channel combination in the config
    cfg.channel_combination = channel_combination
    return
