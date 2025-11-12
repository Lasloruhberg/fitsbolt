# fitsbolt - A Python package for image loading and processing
# Copyright (C) <2025>  <Ruhberg>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the MIT License

import numpy as np


def batch_channel_combination(
    images: np.array, channel_combination: np.ndarray, output_dtype=None
) -> np.ndarray:
    """
    Combine multiple channels with specified weights.
    Will typically return a float array, unless output_dtype is set.

    Args:
        images (np.ndarray): Array of (n_images, H, W, n_extensions)
        channel_combination (np.ndarray): Array of n_output_channels x n_extensions
        original_dtype (optional, np.dtype): Original data type of the images, to enforce

    Returns:
        Combined image array of n_images, H, W, n_output_channels
    """
    # Contract the last axis of images (n_extensions) with the last axis of channel_combination (n_extensions)
    # images: (n_images, H, W, n_extensions) @ channel_combination.T: (n_extensions, n_output_channels)
    # Result: (n_images, H, W, n_output_channels)
    combined = np.tensordot(images, channel_combination.T, axes=([3], [0]))

    if output_dtype is not None and combined.dtype != output_dtype:
        # Apply the same dtype clipping logic as in apply_channel_combination
        if output_dtype == np.uint8:
            combined = np.clip(combined, 0, 255).astype(output_dtype)
        elif output_dtype == np.uint16:
            combined = np.clip(combined, 0, 65535).astype(output_dtype)
        elif output_dtype in [np.int8, np.int16, np.int32, np.int64]:
            info = np.iinfo(output_dtype)
            combined = np.clip(combined, info.min, info.max).astype(output_dtype)
        elif output_dtype in [np.float16, np.float32, np.float64]:
            combined = combined.astype(output_dtype)
        else:
            combined = combined.astype(output_dtype)
    return combined
