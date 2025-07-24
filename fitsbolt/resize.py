import sys
import numpy as np
from skimage.transform import resize
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


from fitsbolt.cfg.create_config import create_config


def resize_images(
    images,
    output_dtype=np.uint8,
    size=None,
    interpolation_order=1,
    desc="Resizing images",
    show_progress=True,
):
    """
    Resize an image to the specified size using skimage's resize function.

    Args:
        images (list(numpy.ndarray)): List of image arrays to resize
        size (tuple, optional): Target size for resizing (height, width). If None, no resizing is done.
        interpolation_order (int, optional): Order of interpolation for resizing with skimage, 0-5. Defaults to 1.

    Returns:
        list(numpy.ndarray): List of resized image arrays

    """
    cfg = create_config(
        output_dtype=output_dtype,
        size=size,
        interpolation_order=interpolation_order,
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
        f"Loading {len(images)} images in parallel with normalisation: {cfg.normalisation_method}"
    )

    def resize_single_image(image):
        try:
            image = _resize_image(
                image,
                cfg,
                convert_to_rgb=True,
            )
            return image
        except Exception as e:
            logger.error(f"Error loading {image}: {str(e)}")
            return None

    # Use ThreadPoolExecutor for parallel loading
    with ThreadPoolExecutor(max_workers=cfg.num_workers) as executor:
        if show_progress:
            results = list(
                tqdm(
                    executor.map(resize_single_image, images),
                    desc=desc,
                    total=len(images),
                )
            )
        else:
            results = list(executor.map(resize_single_image, images))

    # Filter out None results (failed loads)
    results = [r for r in results if r is not None]

    logger.debug(f"Successfully loaded {len(results)} of {len(images)} images")
    return results


def resize_image(image, output_dtype=np.uint8, size=None, interpolation_order=1):
    """
    Resize an image to the specified size using skimage's resize function.

    Args:
        image (numpy.ndarray): Image array to resize
        size (tuple, optional): Target size for resizing (height, width). If None, no resizing is done.
        interpolation_order (int, optional): Order of interpolation for resizing with skimage, 0-5. Defaults to 1.

    Returns:
        numpy.ndarray: Resized image array
    """
    cfg = create_config(
        output_dtype=output_dtype,
        size=size,
        interpolation_order=interpolation_order,
    )

    return _resize_image(image, cfg)


def _resize_image(image, cfg):
    # Simple resize that maintains uint8 type if requested
    if cfg.size is not None and image.shape[:2] != tuple(cfg.size):
        image = resize(
            image,
            cfg.size,
            anti_aliasing=None,
            order=cfg.interpolation_order if cfg.interpolation_order is not None else 1,
            preserve_range=True,
        )
        if cfg.output_dtype == np.uint8:
            image = np.clip(image, 0, np.iinfo(np.uint8).max).astype(np.uint8)
        elif cfg.output_dtype == np.uint16:
            image = np.clip(image, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        elif image.dtype != cfg.output_dtype:
            image = image.astype(cfg.output_dtype)

    return image
