# fitsbolt - A Python package for image loading and processing
# Copyright (C) <2025>  <Ruhberg>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from fitsbolt.cfg.logger import logger


def apply_channel_combination(
    extension_images, channel_combination, original_dtype=None, force_dtype=True
):
    """Applies channel combination to the given extension images.

    Args:
        extension_images (list): list of extension images of shape (H, W), list length = n_fits_extensions.
        channel_combination (numpy.ndarray): Array of channel combination weights (n_output_channels, n_fits_extensions).
        original_dtype (numpy.dtype, optional): Original dtype of the input images.
        force_dtype (bool, optional): If True, forces the output to maintain the original dtype. Defaults to True.

    Returns:
        numpy.ndarray: Combined image (n_output_channels,H, W).
    """
    weights = channel_combination  # Shape: (n_output_channels, n_fits_extensions)

    # Normalize weights to avoid division by zero
    row_sums = np.sum(weights, axis=1, keepdims=True)
    # Replace zero sums with 1 to avoid division by zero
    row_sums[row_sums == 0] = 1
    weights = weights / row_sums

    # Perform dot product along channel dimension
    result = np.tensordot(
        weights, extension_images, axes=1
    )  # n_ouput,n_input x (n_input, H, W) -> (n_output, H, W) ?

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


def convert_greyscale_to_nchannels(image, n_output_channels):
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
            image = np.stack((np.squeeze(image),) * n_output_channels, axis=-1)
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


def batch_channel_combination(
    cutouts: np.array, weights: np.ndarray, original_dtype=None, force_dtype=True
) -> np.ndarray:
    """
    Combine multiple channels with specified weights.

    Args:
        cutouts: Array of n_images, H, W, n_extensions
        weights: Array of n_output_channels x n_extensions

    Returns:
        Combined image array of n_images, H, W, n_output_channels
    """
    # Contract the last axis of cutouts (n_extensions) with the last axis of weights (n_extensions)
    # cutouts: (n_images, H, W, n_extensions) @ weights.T: (n_extensions, n_output_channels)
    # Result: (n_images, H, W, n_output_channels)
    combined = np.tensordot(cutouts, weights.T, axes=([3], [0]))

    if force_dtype and original_dtype is not None and combined.dtype != original_dtype:
        # Apply the same dtype clipping logic as in apply_channel_combination
        if original_dtype == np.uint8:
            combined = np.clip(combined, 0, 255).astype(original_dtype)
        elif original_dtype == np.uint16:
            combined = np.clip(combined, 0, 65535).astype(original_dtype)
        elif original_dtype in [np.int8, np.int16, np.int32, np.int64]:
            info = np.iinfo(original_dtype)
            combined = np.clip(combined, info.min, info.max).astype(original_dtype)
        elif original_dtype in [np.float16, np.float32, np.float64]:
            combined = combined.astype(original_dtype)
        else:
            combined = combined.astype(original_dtype)
    return combined
