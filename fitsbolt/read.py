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
    size=[224, 224],
    fits_extension=None,
    interpolation_order=1,
    n_output_channels=3,
    channel_combination=None,
    num_workers=4,
    desc="Reading images",
    show_progress=True,
):
    """Load and process multiple images in parallel.

    Args:
        filepaths (list): filepath or list of image filepaths to load
        size (list, optional): Target size for image resizing. Defaults to [224, 224].
        fits_extension (int, str, list, optional): The FITS extension(s) to use. Can be:
                                               - An integer index
                                               - A string extension name
                                               - A list of integers or strings to combine multiple extensions
                                               Uses the first extension (0) if None.
        interpolation_order (int, optional): Order of interpolation for resizing with skimage, 0-5. Defaults to 1.
        normalisation_method (NormalisationMethod, optional): Normalisation method to use.
                                                Defaults to NormalisationMethod.CONVERSION_ONLY.
        n_output_channels (int, optional): Number of output channels for the image. Defaults to 3.
        channel_combination (dict, optional): Dictionary defining how to combine FITS extensions into output channels.
                                                Defaults to None, which will try 1:1 or 1:n:output mapping for FITS
        num_workers (int, optional): Number of worker threads for data loading. Defaults to 4.
        desc (str): Description for the progress bar
        show_progress (bool): Whether to show a progress bar

    Returns:
        list: image or list of images for successfully read images
    """

    # check if input is a single filepath or a list
    if not isinstance(filepaths, (list, np.ndarray)):
        return_single = True
        filepaths = [filepaths]

    # create internal configuration object
    cfg = create_config(
        size=size,
        fits_extension=fits_extension,
        interpolation_order=interpolation_order,
        n_output_channels=n_output_channels,
        channel_combination=channel_combination,
        num_workers=num_workers,
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
        else:
            logger.warning(
                "Multiple images loaded but only one was requested. Returning the first image."
            )
            return results[0]
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
                    if cfg.channel_combination is not None:
                        weights = (
                            cfg.channel_combination
                        )  # Shape: (n_output_channels, n_fits_extensions)

                        # Normalize weights
                        weights /= np.sum(weights, axis=1, keepdims=True)
                        # Perform dot product along channel dimension: output shape (H, W, 3)
                        image = np.tensordot(weights, extension_images, axes=1)
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
                else:
                    # Try as string extension name
                    try:
                        image = hdul[fits_extension].data
                    except KeyError:
                        available_ext = [ext.name for ext in hdul if hasattr(ext, "name")]
                        logger.error(
                            f"FITS extension name '{fits_extension}' not found in file {filepath}. "
                            + f"Available extensions: {available_ext}"
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
            # Normalisation is done later
            if image.dtype != np.uint8:
                # Safe normalization that handles edge cases
                img_min, img_max = image.min(), image.max()
                if img_max <= img_min:
                    # incorrect image, set to zero
                    logger.warning(
                        f"FITS image {filepath} has no valid data (min=max). Setting to zero."
                    )
                    image = np.zeros_like(image, dtype=np.uint8)

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

    return image
