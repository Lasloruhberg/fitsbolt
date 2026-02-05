# fitsbolt - A Python package for image loading and processing
# Copyright (C) <2025>  <Ruhberg>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the MIT or GPL-3.0 License

import numpy as np


def _convert_dtype(data, output_dtype):
    """Convert array to output_dtype with appropriate clipping."""
    if output_dtype is None or data.dtype == output_dtype:
        return data
    if output_dtype == np.uint8:
        return np.clip(data, 0, 255).astype(output_dtype)
    elif output_dtype == np.uint16:
        return np.clip(data, 0, 65535).astype(output_dtype)
    elif output_dtype in [np.int8, np.int16, np.int32, np.int64]:
        info = np.iinfo(output_dtype)
        return np.clip(data, info.min, info.max).astype(output_dtype)
    else:
        return data.astype(output_dtype)


def _is_slicing_matrix(cc):
    """Check if channel_combination is eye(n_out, n_in) — i.e. just selecting
    the first n_out channels.  Returns n_out if true, else 0."""
    n_out, n_in = cc.shape
    if n_out > n_in:
        return 0
    expected = np.eye(n_out, n_in, dtype=cc.dtype)
    if np.array_equal(cc, expected):
        return n_out
    return 0


def _is_broadcast_matrix(cc):
    """Check if channel_combination is ones(n_out, 1) — broadcasting a single
    channel to n_out identical channels.  Returns n_out if true, else 0."""
    n_out, n_in = cc.shape
    if n_in != 1:
        return 0
    if np.all(cc == 1.0):
        return n_out
    return 0


def batch_channel_combination(
    images: np.array, channel_combination: np.ndarray, output_dtype=None
) -> np.ndarray:
    """
    Combine multiple channels with specified weights.
    Will typically return a float array, unless output_dtype is set.

    Includes fast paths for common cases (identity, channel slicing,
    single-channel broadcast) to avoid expensive tensordot operations.

    Args:
        images (np.ndarray): Array of (n_images, H, W, n_extensions)
        channel_combination (np.ndarray): Array of n_output_channels x n_extensions
        output_dtype (optional, np.dtype): Original data type of the images, to enforce

    Returns:
        Combined image array of n_images, H, W, n_output_channels
    """
    cc = np.asarray(channel_combination, dtype=np.float64)
    n_out, n_in = cc.shape

    # --- Fast path: slicing / identity matrix (eye(n_out, n_in)) ---
    slice_n = _is_slicing_matrix(cc)
    if slice_n:
        if slice_n == n_in:
            # Full identity — no channel change needed
            return _convert_dtype(images, output_dtype)
        # Slice first n_out channels (e.g. RGBA → RGB)
        combined = np.ascontiguousarray(images[..., :slice_n])
        return _convert_dtype(combined, output_dtype)

    # --- Fast path: broadcast single channel to n_out ---
    broadcast_n = _is_broadcast_matrix(cc)
    if broadcast_n:
        combined = np.repeat(images, broadcast_n, axis=-1)
        return _convert_dtype(combined, output_dtype)

    # --- General case: tensordot ---
    # Contract the last axis of images (n_extensions) with the last axis of channel_combination (n_extensions)
    # images: (n_images, H, W, n_extensions) @ channel_combination.T: (n_extensions, n_output_channels)
    # Result: (n_images, H, W, n_output_channels)
    combined = np.tensordot(images, cc.T, axes=([3], [0]))
    return _convert_dtype(combined, output_dtype)
