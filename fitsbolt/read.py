import os
import sys
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from loguru import logger
from astropy.io import fits

from fitsbolt.cfg.create_config import create_config
from fitsbolt.cfg.create_config import SUPPORTED_IMAGE_EXTENSIONS


def read_images(
    filepaths,
    fits_extension=None,
    n_output_channels=3,
    channel_combination=None,
    num_workers=4,
    desc="Reading images",
    show_progress=True,
    force_dtype=True,
):
    """Load and process multiple images in parallel.

    Args:
        filepaths (list): filepath or list of image filepaths to load
        fits_extension (int, str, list, optional): The FITS extension(s) to use. Can be:
                                               - An integer index
                                               - A string extension name
                                               - A list of integers or strings to combine multiple extensions
                                               Uses the first extension (0) if None.
        normalisation_method (NormalisationMethod, optional): Normalisation method to use.
                                                Defaults to NormalisationMethod.CONVERSION_ONLY.
        n_output_channels (int, optional): Number of output channels for the image. Defaults to 3.
        channel_combination (dict, optional): Dictionary defining how to combine FITS extensions into output channels.
                                                Defaults to None, which will try 1:1 or 1:n:output mapping for FITS
        num_workers (int, optional): Number of worker threads for data loading. Defaults to 4.
        desc (str): Description for the progress bar
        show_progress (bool): Whether to show a progress bar
        force_dtype (bool, optional): If True, forces the output to maintain the original dtype after tensor operations
                                     like channel combination. Defaults to True.

    Returns:
        list: image or list of images for successfully read images
    """

    # check if input is a single filepath or a list
    if not isinstance(filepaths, (list, np.ndarray)):
        return_single = True
        filepaths = [filepaths]
    else:
        return_single = False

    # create internal configuration object
    cfg = create_config(
        fits_extension=fits_extension,
        n_output_channels=n_output_channels,
        channel_combination=channel_combination,
        num_workers=num_workers,
        force_dtype=force_dtype,
    )

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

    def read_single_image(filepath):
        try:
            image = _read_image(
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
                    executor.map(read_single_image, filepaths),
                    desc=desc,
                    total=len(filepaths),
                )
            )
        else:
            results = list(executor.map(read_single_image, filepaths))

    # Filter out None results (failed loads)
    results = [r for r in results if r is not None]

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
            return None
    return results


def _read_image(filepath, cfg):
    """
    Read image data from a file without processing.

    Args:
        filepath (str): Path to the image file
        cfg: internal configuration object

    Returns:
        numpy.ndarray: Raw image array
    """
    fits_extension = cfg.fits_extension
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
                            f"Single extension should be 2D. Replacing with black image."
                        )
                        # Create a 2D black image with the spatial dimensions of the original
                        image = np.zeros((image.shape[1], image.shape[2]), dtype=image.dtype)
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
                                f"Setting to empty image"
                            )
                            # use dim 1 as in both H,W,C or C,H,W this will work for square images
                            ext_data = np.zeros(
                                (ext_data.shape[1], ext_data.shape[1]), dtype=ext_data.dtype
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

                    # Combine the extensions into n_output channels
                    # Do a linear combination based on the configuration
                    original_dtype = extension_images[0].dtype if extension_images else None
                    if cfg.channel_combination is not None:
                        image = _apply_channel_combination(
                            extension_images,
                            cfg.channel_combination,
                            original_dtype,
                            cfg.force_dtype,
                        )
                    else:
                        # Stack the extensions along a new dimension
                        image = np.stack(extension_images)

                    # If images are 2D (Height, Width), stack results in 3D array (Ext, Height, Width)
                    # If images are 3D (Height, Width, Channels), stack results in 4D (Ext, Height, Width, Channels)
                    # For 2D images (now 3D after stacking), treat extensions as channels (RGB)
                    if len(extension_shapes[0]) == 2:
                        # Only use up to 3 extensions for RGB (more will be handled later by truncation)
                        if len(image) > 3:
                            import warnings

                            warnings.warn(
                                f"More than 3 extensions provided for file {filepath}. "
                                f"Only the first 3 will be used as RGB channels."
                            )
                            logger.warning(
                                f"More than 3 extensions provided for file {filepath}. "
                                f"Only the first 3 will be used as RGB channels."
                            )
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
                            f"Single extension should be 2D. Replacing with black image."
                        )
                        # Create a 2D black image with the spatial dimensions of the original
                        image = np.zeros((image.shape[1], image.shape[2]), dtype=image.dtype)
                else:
                    # Try as string extension name
                    try:
                        image = hdul[fits_extension].data

                        # Check if the loaded data is 3D when we expect 2D (single extension)
                        if image.ndim == 3:
                            logger.warning(
                                f"FITS extension '{fits_extension}' in file {filepath} contains 3D data "
                                f"(shape: {image.shape}). Single extension should be 2D. Replacing with black image."
                            )
                            # Create a 2D black image with the spatial dimensions of the original
                            image = np.zeros((image.shape[1], image.shape[2]), dtype=image.dtype)
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
                logger.warning(
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

    # check if there is a greyscale image or an RGBA image that needs to be converted to RGB
    image = _convert_greyscale_to_nchannels(
        image,
        n_output_channels=cfg.n_output_channels,
    )

    # Validate output shape based on n_output_channels
    if cfg.n_output_channels == 1:
        assert (
            len(image.shape) == 2
        ), f"After reading, single channel image has unexpected shape: {image.shape} - {filepath}"
    else:
        assert (
            len(image.shape) == 3 and image.shape[2] == cfg.n_output_channels
        ), f"After reading, image has unexpected shape: {image.shape} - {filepath}"

    # return H,W for single channel or H,W,C for multi-channel
    return image


def _apply_channel_combination(
    extension_images, channel_combination, original_dtype=None, force_dtype=True
):
    """Applies channel combination to the given extension images.

    Args:
        extension_images (list): list of extension images of shape (H, W), list length = n_fits_extensions.
        channel_combination (numpy.ndarray): Array of channel combination weights (n_output_channels, n_fits_extensions).
        original_dtype (numpy.dtype, optional): Original dtype of the input images.
        force_dtype (bool, optional): If True, forces the output to maintain the original dtype. Defaults to True.

    Returns:
        numpy.ndarray: Combined image (H, W, n_output_channels).
    """
    weights = channel_combination  # Shape: (n_output_channels, n_fits_extensions)

    # Normalize weights
    weights /= np.sum(weights, axis=1, keepdims=True)
    # Perform dot product along channel dimension: output shape (H, W, n_output_channels)
    result = np.tensordot(weights, extension_images, axes=1)

    # Force dtype if requested and original dtype is provided
    if force_dtype and original_dtype is not None and result.dtype != original_dtype:
        # Ensure the result stays within the valid range for the target dtype
        if original_dtype == np.uint8:
            result = np.clip(result, 0, 255).astype(original_dtype)
        elif original_dtype == np.uint16:
            result = np.clip(result, 0, 65535).astype(original_dtype)
        elif original_dtype in [np.int8, np.int16, np.int32, np.int64]:
            info = np.iinfo(original_dtype)
            result = np.clip(result, info.min, info.max).astype(original_dtype)
        elif original_dtype in [np.float16, np.float32, np.float64]:
            result = result.astype(original_dtype)
        else:
            result = result.astype(original_dtype)

    return result


def _convert_greyscale_to_nchannels(image, n_output_channels):
    """
    Convert a grayscale image to a specified number of channels.
    This function provides some backwards compatibility
    Args:
        image (numpy.ndarray): Grayscale image array (H,W,C or H,W)
        n_output_channels (int): Number of output channels
    Returns:
        numpy.ndarray: Image with the specified number of channels (H,W,n_output_channels)
    """

    # Handle grayscale images
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        if n_output_channels != 1:
            image = np.stack((image,) * n_output_channels, axis=-1)
        else:
            # always return a 2D image if n_output_channels is 1
            image = image[:, :, 0] if len(image.shape) == 3 else image

    # if e.g. rgb (n_output_channels == 3) is requested but an rgba png is loaded
    # Handle e.g. RGBA images (for png, tiff support)
    if len(image.shape) == 3 and image.shape[2] > n_output_channels:
        logger.trace(
            "Image is in RGBA format. Converting to RGB by dropping the excess (e.g. alpha) channels."
        )
        image = image[:, :, :n_output_channels]
    return image
