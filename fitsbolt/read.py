# fitsbolt - A Python package for image loading and processing
# Copyright (C) <2025>  <Ruhberg>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the MIT License

import os
import numpy as np
import warnings
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from astropy.io import fits

from .cfg.create_config import (
    create_config,
    SUPPORTED_IMAGE_EXTENSIONS,
    recompute_config_channel_combination,
)
from .cfg.logger import logger
from .channel_mixing import (
    batch_channel_combination,
)


def read_images(
    filepaths,
    fits_extension=None,
    n_output_channels=3,
    channel_combination=None,
    num_workers=4,
    desc="Reading images",
    show_progress=True,
    force_dtype=True,
    log_level="WARNING",
    read_only=False,
):
    """Load and process multiple images in parallel.

    Args:
        filepaths (list): filepath or list of image filepaths to load, or list of lists for multi-FITS mode
        fits_extension (int, str, list, optional): The FITS extension(s) to use. Can be:
                                               - An integer index
                                               - A string extension name
                                               - A list of integers or strings to combine multiple extensions
                                               - For multi-FITS mode: list of extensions matching filepaths structure
                                               Uses the first extension (0) if None.
        n_output_channels (int, optional): Number of output channels for the image. Defaults to 3.
        channel_combination (dict, optional): Dictionary defining how to combine FITS extensions into output channels.
                                                Defaults to None, which will try 1:1 or 1:n:output mapping for FITS
        num_workers (int, optional): Number of worker threads for data loading. Defaults to 4.
        desc (str): Description for the progress bar
        show_progress (bool): Whether to show a progress bar
        force_dtype (bool, optional): If True, forces the output to maintain the original dtype after tensor operations
                                     like channel combination. Defaults to True.
        log_level (str, optional): Logging level for the operation. Defaults to "WARNING".
                                   Can be "TRACE", "DEBUG", "INFO", "WARNING", "ERROR", or "CRITICAL".
        read_only (bool, optional): If True, skips the channel combination logic.

    Returns:
        list: image or list of images for successfully read images

    Examples:
        # Single FITS file with multiple extensions
        images = read_images(["image.fits"], fits_extension=[0, 1, 2])

        # Multiple FITS files with corresponding extensions (new functionality)
        images = read_images([["file1.fits", "file2.fits", "file3.fits"]],
                           fits_extension=[0, 1, 2])
    """
    if read_only:
        if fits_extension is not None:
            if isinstance(filepaths[0], list):
                n_output_channels = len(filepaths[0])
            else:
                if isinstance(fits_extension, list):
                    n_output_channels = len(fits_extension)
                else:
                    n_output_channels = 1

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
        if not len(fits_extension) == len(filepaths[0]):
            raise ValueError("Multi-FITS mode requires fits_extension to match the number of files")
        multi_fits_mode = True
    else:
        multi_fits_mode = False

    # create internal configuration object
    cfg = create_config(
        fits_extension=fits_extension,
        n_output_channels=n_output_channels,
        channel_combination=channel_combination,
        num_workers=num_workers,
        force_dtype=force_dtype,
        log_level=log_level,
    )

    # Add a new logger configuration for console output
    logger.set_log_level(cfg.log_level)

    logger.debug(f"Setting LogLevel to {cfg.log_level.upper()}")

    logger.debug(
        f"Loading {len(filepaths)} images in parallel with normalisation: {cfg.normalisation_method}"
    )
    if multi_fits_mode:

        def read_single_image(filepaths):
            try:
                image = _read_multi_fits_image(
                    filepaths,
                    fits_extension,
                    cfg,
                )
                return image
            except Exception as e:
                logger.error(f"Error loading {filepaths}: {str(e)}")
                raise e

    else:

        def read_single_image(filepath):
            try:
                image = _read_image(
                    filepath,
                    cfg,
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
                    executor.map(read_single_image, filepaths),
                    desc=desc,
                    total=len(filepaths),
                )
            )
        else:
            results = list(executor.map(read_single_image, filepaths))

    # Combine channels
    # Do a linear combination based on the configuration
    # this is only necessary for inputs where fits_extension is None,
    # otherwise create_cfg will have managed it
    channel_combination_exists = cfg.get("channel_combination") is not None
    if not channel_combination_exists:
        recompute_config_channel_combination(cfg)

    original_dtype = results[0].dtype
    if cfg.channel_combination is not None and not read_only:
        results = batch_channel_combination(
            results,
            cfg.channel_combination,
            output_dtype=original_dtype,
        )

    logger.debug(f"Successfully loaded {len(results)} of {len(filepaths)} images")
    if return_single:
        # If only one image was requested, return it directly
        if len(results) == 1:
            return results[0]
        elif len(results) > 1:
            logger.warning(
                "Multiple images loaded but only one was requested. Returning the first image."
            )
            return results[0]
        else:
            logger.error("No images were successfully loaded")
            raise ValueError("No images were successfully loaded")
    return results


def _read_multi_fits_image(filepaths, fits_extensions, cfg):
    """
    Read and combine multiple FITS files with corresponding extensions.

    Args:
        filepaths (list): List of FITS file paths
        fits_extensions (list): List of extensions corresponding to each file
        cfg: internal configuration object

    Returns:
        numpy.ndarray: Combined image array
    """
    # Validate input lengths match
    assert len(filepaths) == len(fits_extensions), (
        f"Number of FITS files ({len(filepaths)}) must match number of extensions "
        f"({len(fits_extensions)}). Files: {filepaths}, Extensions: {fits_extensions}"
    )

    # Read each FITS file with its corresponding extension
    extension_images = []
    extension_shapes = []
    extension_names = []

    for filepath, extension in zip(filepaths, fits_extensions):
        # Validate file is a FITS file
        file_ext = os.path.splitext(filepath.lower())[1]
        assert file_ext == ".fits", (
            f"All files must be FITS files when using multi-file mode. "
            f"Got {file_ext} for file {filepath}"
        )

        logger.trace(f"Reading FITS file {filepath} with extension {extension}")

        with fits.open(filepath) as hdul:
            # Handle extension access
            if isinstance(extension, (int, np.integer)):
                extension_idx = int(extension)
                if extension_idx < 0 or extension_idx >= len(hdul):
                    logger.error(
                        f"Invalid FITS extension index {extension_idx} for file {filepath} "
                        f"with {len(hdul)} extensions"
                    )
                    raise IndexError(
                        f"FITS extension index {extension_idx} is out of bounds (0-{len(hdul) - 1})"
                    )
                ext_data = hdul[extension_idx].data
                extension_names.append(f"{filepath}[{extension_idx}]")
            else:
                # Try as string extension name
                try:
                    ext_data = hdul[extension].data
                    extension_names.append(f"{filepath}['{extension}']")
                except KeyError:
                    available_ext = [ext.name for ext in hdul if hasattr(ext, "name")]
                    logger.error(
                        f"FITS extension name '{extension}' not found in file {filepath}. "
                        f"Available extensions: {available_ext}"
                    )
                    raise KeyError(f"FITS extension name '{extension}' not found")

            # Check for None data
            if ext_data is None:
                logger.error(f"FITS extension {extension} in file {filepath} has no data")
                raise ValueError(f"FITS extension {extension} in file {filepath} has no data")

            # Handle dimension issues
            if ext_data.ndim > 2:
                logger.warning(
                    f"FITS extension {extension} in file {filepath} has more than 2 dimensions. "
                )
                raise ValueError(
                    f"FITS extension {extension} in file {filepath} has more than 2 dimensions. "
                    "Not supported"
                )

            extension_images.append(ext_data)
            extension_shapes.append(ext_data.shape)

    # Validate all shapes match
    if len(set(str(shape) for shape in extension_shapes)) > 1:
        shape_info = [f"{name}: {shape}" for name, shape in zip(extension_names, extension_shapes)]
        error_msg = (
            f"Cannot combine FITS files with different shapes. "
            f"Extension shapes: {', '.join(shape_info)}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Combine the images into n_output channels (removed to other function)
    # Stack the extensions along a new dimension
    image = np.stack(extension_images)

    # For 2D images (now 3D after stacking), treat extensions as channels (RGB)
    if len(extension_shapes[0]) == 2:
        # Only use up to 3 extensions for RGB (more will be handled later by truncation)
        if len(image) > 3:
            import warnings

            warnings.warn(
                "More than 3 FITS files provided. " "Only the first 3 will be used as RGB channels."
            )
            logger.warning(
                "More than 3 FITS files provided. " "Only the first 3 will be used as RGB channels."
            )
        # Transpose to get (Height, Width, Extensions) which is compatible with RGB format
        image = np.transpose(image, (1, 2, 0))
    # this is now more difficult, check if fits should be loaded and check if there is a mismatch
    if not cfg.get("n_expected_channels"):
        cfg.n_expected_channels = len(filepaths)

    if image is None:
        logger.error(f"Failed to read image from {filepath}")
        raise ValueError(f"Image reading failed for {filepath}. Check the file format and content.")
    elif isinstance(image, np.ndarray):
        if image.size == 0:
            logger.error(f"Image from {filepath} is empty (size 0)")
            raise ValueError(f"Image from {filepath} is empty (size 0)")
    return image


def _read_image(filepath, cfg):
    """
    Read image data from a file without processing.

    Args:
        filepath (str or list): Path to the image file, or list of FITS file paths for multi-file mode
        cfg: internal configuration object

    Returns:
        numpy.ndarray: Raw image array
    """
    fits_extension = cfg.fits_extension

    # Handle multi-file FITS input
    if isinstance(filepath, list):
        # Multi-file mode: filepath is a list of FITS files, fits_extension should be a list too
        if not isinstance(fits_extension, list):
            raise ValueError(
                f"When providing multiple FITS files, fits_extension must be a list. "
                f"Got {type(fits_extension)}: {fits_extension}"
            )
        return _read_multi_fits_image(filepath, fits_extension, cfg)

    # Single file mode (original functionality)
    # Get file extension
    file_ext = os.path.splitext(filepath.lower())[1]

    # Validate file extension
    assert file_ext in SUPPORTED_IMAGE_EXTENSIONS, (
        f"Unsupported file extension {file_ext} for file {filepath}. "
        f"Supported extensions: {SUPPORTED_IMAGE_EXTENSIONS}"
    )
    logger.trace(f"Reading image {filepath} with extension {file_ext}")

    if file_ext == ".fits":
        # Handle FITS files with astropy
        with fits.open(filepath) as hdul:
            try:
                # Handle different extension types (None, int, string, or list)
                if fits_extension is None:
                    # Default to first extension (index 0)
                    image = hdul[0].data

                    # Check if the loaded data is 3D when we expect 2D (single extension)
                    if image.ndim == 3:
                        logger.warning(
                            f"FITS extension 0 in file {filepath} contains 3D data (shape: {image.shape}). "
                            f"Single extension should be 2D."
                        )
                        # Create a 2D black image with the spatial dimensions of the original
                        raise ValueError(
                            f"FITS extension 0 in file {filepath} contains 3D data (shape: {image.shape}). "
                            f"Single extension should be 2D. Not supported"
                        )

                elif isinstance(fits_extension, list):
                    # Handle list of extensions - need to load and combine them
                    extension_images = []
                    extension_shapes = []
                    extension_names = []

                    # First load all extensions to validate shapes match
                    for ext in fits_extension:
                        if isinstance(ext, (int, np.integer)):
                            # Integer index - check valid bounds
                            ext_idx = int(ext)
                            if ext_idx < 0 or ext_idx >= len(hdul):
                                available_indices = list(range(len(hdul)))
                                logger.error(
                                    f"Invalid FITS extension index {ext_idx} for file {filepath}. "
                                    f"Available indices: {available_indices}"
                                )
                                raise IndexError(
                                    f"FITS extension index {ext_idx} is out of bounds (0-{len(hdul) - 1})"
                                )
                            ext_data = hdul[ext_idx].data
                            extension_names.append(f"extension {ext_idx}")
                        else:
                            # Try as string extension name
                            try:
                                ext_data = hdul[ext].data
                                extension_names.append(f"'{ext}'")
                            except KeyError:
                                available_ext = [
                                    ext_name.name for ext_name in hdul if hasattr(ext_name, "name")
                                ]
                                logger.error(
                                    f"FITS extension name '{ext}' not found in file {filepath}. "
                                    f"Available extensions: {available_ext}"
                                )
                                raise KeyError(f"FITS extension name '{ext}' not found")

                        # Check for None data
                        if ext_data is None:
                            logger.error(f"FITS extension {ext} in file {filepath} has no data")
                            raise ValueError(f"FITS extension {ext} in file {filepath} has no data")

                        # Record the shape for validation
                        if ext_data.ndim > 2:
                            logger.warning(
                                f"FITS extension {ext} in file {filepath} has more than 2 dimensions. "
                                "Not supported"
                            )
                            # use dim 1 as in both H,W,C or C,H,W this will work for square images
                            raise ValueError(
                                f"FITS extension {ext} in file {filepath} has more than 2 dimensions. "
                                "Not supported"
                            )
                        extension_images.append(ext_data)
                        extension_shapes.append(ext_data.shape)

                    # Validate all shapes match
                    if len(set(str(shape) for shape in extension_shapes)) > 1:
                        shape_info = [
                            f"{name}: {shape}"
                            for name, shape in zip(extension_names, extension_shapes)
                        ]
                        error_msg = (
                            f"Cannot combine FITS extensions with different shapes in file {filepath}. "
                            f"Extension shapes: {', '.join(shape_info)}"
                        )
                        logger.error(error_msg)
                        raise ValueError(error_msg)

                    # Combine the extensions into n_output channels (removed to other function)
                    # Stack the extensions along a new dimension
                    image = np.stack(extension_images)

                    # If images are 2D (Height, Width), stack results in 3D array (Ext, Height, Width)
                    # If images are 3D (Height, Width, Channels), stack results in 4D (Ext, Height, Width, Channels)
                    # For 2D images (now 3D after stacking), treat extensions as channels (RGB)
                    if len(extension_shapes[0]) == 2:
                        # Transpose to get (Height, Width, Extensions) which is compatible with RGB format
                        image = np.transpose(image, (1, 2, 0))
                elif isinstance(fits_extension, (int, np.integer)):
                    # Integer index - check valid bounds
                    extension_idx = int(fits_extension)
                    if extension_idx < 0 or extension_idx >= len(hdul):
                        logger.error(
                            f"Invalid FITS extension index {extension_idx} for file {filepath} with {len(hdul)} extensions"
                        )
                        raise IndexError(
                            f"FITS extension index {extension_idx} is out of bounds (0-{len(hdul) - 1})"
                        )
                    image = hdul[extension_idx].data

                    # Check if the loaded data is 3D when we expect 2D (single extension)
                    if image.ndim == 3:
                        logger.warning(
                            f"FITS extension {extension_idx} in file {filepath} contains 3D data (shape: {image.shape}). "
                            f"Single extension should be 2D. Not supported"
                        )
                        raise ValueError(
                            f"FITS extension {extension_idx} in file {filepath} contains 3D data (shape: {image.shape}). "
                            f"Single extension should be 2D. Not supported"
                        )
                else:
                    # Try as string extension name
                    try:
                        image = hdul[fits_extension].data

                        # Check if the loaded data is 3D when we expect 2D (single extension)
                        if image.ndim == 3:
                            logger.warning(
                                f"FITS extension '{fits_extension}' in file {filepath} contains 3D data "
                                f"(shape: {image.shape}). Single extension should be 2D."
                            )
                            # Create a 2D black image with the spatial dimensions of the original
                            raise ValueError(
                                f"FITS extension '{fits_extension}' in file {filepath} contains 3D data "
                                f"(shape: {image.shape}). Single extension should be 2D. Not supported"
                            )
                    except KeyError:
                        available_ext = [ext.name for ext in hdul if hasattr(ext, "name")]
                        logger.error(
                            f"FITS extension name '{fits_extension}' not found in file {filepath}. "
                            f"Available extensions: {available_ext}"
                        )
                        raise KeyError(f"FITS extension name '{fits_extension}' not found")
            except Exception as e:
                if isinstance(e, (IndexError, KeyError, ValueError)):
                    # Re-raise specific extension errors
                    raise
                else:
                    # For other errors, log and re-raise
                    logger.error(
                        f"Error accessing FITS extension {fits_extension} in file {filepath}: {e}"
                    )
                    raise

            # Handle case where data is None
            if image is None:
                logger.error(f"FITS extension {fits_extension} in file {filepath} has no data")
                raise ValueError(f"FITS extension {fits_extension} in file {filepath} has no data")

            # Handle dimension issues in FITS data
            if image.ndim > 3:
                warnings.warn(
                    f"FITS image {filepath} has more than 3 dimensions. Taking the first 3 dimensions."
                )
                image = image[:3]
                if image.shape[0] < image.shape[-1]:
                    logger.warning(
                        f"FITS image {filepath} seems to be in Channel x Height x Width format. Transposing."
                    )
                    image = np.transpose(image, (1, 2, 0))
            # Validate that we have a valid image with at least 2 dimensions
            assert (
                image.ndim >= 2 and image.ndim <= 3
            ), f"FITS image {filepath} has less than 2 or more than 3 dimensions: {image.shape}"
    else:
        # Use PIL for standard image formats
        image = np.array(Image.open(filepath))

        # Validate the image has appropriate dimensions
        assert (
            image.ndim >= 2 and image.ndim <= 3
        ), f"Image {filepath} has less than 2 or more than 3 dimensions: {image.shape}"

        # raise an error if the image is empty or not loaded correctly
        assert image.size > 0, f"Image {filepath} is empty or not loaded correctly"

    # this is now more difficult, check if fits should be loaded and check if there is a mismatch
    if not cfg.get("n_expected_channels"):
        # no previous image read, try to gather what the value should be
        if cfg.fits_extension is not None:
            if isinstance(cfg.fits_extension, list):
                n_expected_channels = [len(cfg.fits_extension)]
            else:
                # Single extension (int or string), expect 1 channel
                n_expected_channels = [1]
        else:
            n_expected_channels = [1, 3, 4]  # RGB or RGBA
    else:
        # already read an image and fixed n_expected channels
        n_expected_channels = cfg.n_expected_channels

    # make sure image is in H,W,C and not C,H,W format
    if image.shape[0] in n_expected_channels and image.shape[-1] not in n_expected_channels:
        image = np.transpose(image, (1, 2, 0))

    # Check if the image has the expected number of channels
    if len(image.shape) == 2:
        # Grayscale image (1 channel)
        assert 1 in n_expected_channels, (
            "Unexpected number of channels: 1 (grayscale),"
            + f"expected one of {n_expected_channels} from other files/ fits_extension - {filepath}"
        )
        # If the image is 2D and we expect 1 channel, we can add a channel dimension to make H,W,1
        image = image[..., np.newaxis]
    else:
        # Multi-channel image
        assert image.shape[2] in n_expected_channels, (
            f"Unexpected number of channels: {image.shape[2]},"
            + f"expected one of {n_expected_channels} from other files/ fits_extension - {filepath}"
        )
    if not cfg.get("n_expected_channels"):
        cfg.n_expected_channels = [image.shape[2]]
    # return H,W for single channel or H,W,C for multi-channel
    if image is None:
        logger.error(f"Failed to read image from {filepath}")
        raise ValueError(
            f"Image reading failed for {filepath}." + "Check the file format and content."
        )
    elif isinstance(image, np.ndarray):
        if image.size == 0:
            logger.error(f"Image from {filepath} is empty (size 0)")
            raise ValueError(f"Image from {filepath} is empty (size 0)")
    return image
