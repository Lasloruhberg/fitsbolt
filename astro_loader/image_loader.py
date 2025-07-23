import os
import sys
import numpy as np
from PIL import Image
from skimage.transform import resize
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from loguru import logger

from astropy.io import fits

from astro_loader.normalisation.NormalisationMethod import NormalisationMethod
from astro_loader.normalisation.normalisation import normalise_image
from astro_loader.cfg.create_config import create_config, validate_config


SUPPORTED_IMAGE_EXTENSIONS = {".fits", ".jpg", ".jpeg", ".png", ".tiff"}


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


def process_image(
    image,
    cfg,
    convert_to_rgb=True,
    image_source="array",
):
    """
    Process an image array by normalising and resizing it.
    Args:
        image (numpy.ndarray): Image array to process
        cfg: Configuration object containing size, normalisation_method
        convert_to_rgb (bool): Whether to convert grayscale/RGBA to RGB
        image_source (str): Source of the image for logging
    Returns:
        numpy.ndarray: Processed image array as uint8
    """
    try:

        # Convert to RGB if requested
        if convert_to_rgb:
            # Handle grayscale images
            if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
                image = np.stack((image,) * 3, axis=-1)
            # Handle RGBA images
            elif len(image.shape) == 3 and image.shape[2] > 3:
                logger.trace(
                    f"Image {image_source} is in RGBA format. Converting to RGB by dropping the alpha channel."
                )
                image = image[:, :, :3]

            # Validate RGB structure after conversion
            if convert_to_rgb:
                assert (
                    len(image.shape) == 3 and image.shape[2] == 3
                ), f"After RGB conversion, image {image_source} has unexpected shape: {image.shape}"

        logger.trace(f"Normalising image with setting {cfg.normalisation_method}")
        image = normalise_image(image, cfg=cfg)

        # Simple resize that maintains uint8 type if requested
        if cfg.size is not None and image.shape[:2] != tuple(cfg.size):
            image = resize(
                image,
                cfg.size,
                anti_aliasing=None,
                order=cfg.interpolation_order if cfg.interpolation_order is not None else 1,
                preserve_range=True,
            )
            image = np.clip(image, 0, 255).astype(np.uint8)

        return image

    except Exception as e:
        logger.error(f"Error processing image {image_source}: {e}")
        raise e


def _load_image(filepath, cfg, convert_to_rgb=True):
    try:
        # Read raw image data
        image = _read_image(filepath, cfg)

        # Process the image using the centralized processing function
        return process_image(image, cfg, convert_to_rgb=convert_to_rgb, image_source=filepath)

    except Exception as e:
        logger.error(f"Error reading image {filepath}: {e}")
        raise e


def load_and_process_images(
    filepaths,
    cfg=None,
    output_dtype=np.uint8,
    size=[224, 224],
    fits_extension=None,
    interpolation_order=1,
    normalisation_method=NormalisationMethod.CONVERSION_ONLY,
    channel_combination=None,
    num_workers=4,
    norm_maximum_value=None,
    norm_minimum_value=None,
    norm_log_calculate_minimum_value=False,
    norm_crop_for_maximum_value=None,
    norm_asinh_scale=[0.7, 0.7, 0.7],
    norm_asinh_clip=[99.8, 99.8, 99.8],
    desc="Loading images",
    show_progress=True,
):
    """Load and process multiple images in parallel.

    Args:
        filepaths (list): List of image filepaths to load
        cfg (DotMap, optional): Configuration settings. Defaults to None.
        output_dtype (type, optional): Data type for output images. Defaults to np.uint8.
        size (list, optional): Target size for image resizing. Defaults to [224, 224].
        fits_extension (int, str, list, optional): The FITS extension(s) to use. Can be:
                                               - An integer index
                                               - A string extension name
                                               - A list of integers or strings to combine multiple extensions
                                               Uses the first extension (0) if None.
        interpolation_order (int, optional): Order of interpolation for resizing with skimage, 0-5. Defaults to 1.
        normalisation_method (NormalisationMethod, optional): Normalisation method to use.
                                                Defaults to NormalisationMethod.CONVERSION_ONLY.
        channel_combination (dict, optional): Dictionary defining how to combine FITS extensions into output channels.
                                                Defaults to None.
        num_workers (int, optional): Number of worker threads for data loading. Defaults to 4.
        norm_maximum_value (float, optional): Maximum value for normalisation. Defaults to None.
        norm_minimum_value (float, optional): Minimum value for normalisation. Defaults to None.
        norm_log_calculate_minimum_value (bool, optional): If True, calculates the minimum value when log scaling
                                                (normally defaults to 0). Defaults to False.
        norm_crop_for_maximum_value (tuple, optional): Crops the image for maximum value. Defaults to None.
        norm_asinh_scale (list, optional): Scale factors for asinh normalisation. Defaults to [0.7, 0.7, 0.7].
        norm_asinh_clip (list, optional): Clip values for asinh normalisation. Defaults to [99.8, 99.8, 99.8].
        filepaths (list): List of image filepaths to load
        size (tuple, optional): Size to resize images to (height, width)
        desc (str): Description for the progress bar
        show_progress (bool): Whether to show a progress bar

    Returns:
        list: List of (filepath, image) tuples for successfully loaded images
    """

    if cfg is None:
        cfg = create_config(
            output_dtype=output_dtype,
            size=size,
            fits_extension=fits_extension,
            interpolation_order=interpolation_order,
            normalisation_method=normalisation_method,
            channel_combination=channel_combination,
            num_workers=num_workers,
            norm_maximum_value=norm_maximum_value,
            norm_minimum_value=norm_minimum_value,
            norm_log_calculate_minimum_value=norm_log_calculate_minimum_value,
            norm_crop_for_maximum_value=norm_crop_for_maximum_value,
            norm_asinh_scale=norm_asinh_scale,
            norm_asinh_clip=norm_asinh_clip,
        )
    else:
        validate_config(cfg)
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

    def load_single_image(filepath):
        try:
            image = _load_image(
                filepath,
                cfg,
                convert_to_rgb=True,
            )
            return filepath, image
        except Exception as e:
            logger.error(f"Error loading {filepath}: {str(e)}")
            return None

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

    # Filter out None results (failed loads)
    results = [r for r in results if r is not None]

    logger.debug(f"Successfully loaded {len(results)} of {len(filepaths)} images")
    return results
