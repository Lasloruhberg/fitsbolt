# fitsbolt - A Python package for image loading and processing
# Copyright (C) <2025>  <Ruhberg>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the MIT License

"""
Tests for the normalisation functionality in normalisation.py and NormalisationMethod.py.
"""

import numpy as np
import pytest
import logging

from fitsbolt.cfg.create_config import create_config
from fitsbolt.cfg import logger
from fitsbolt.normalisation.normalisation import (
    _normalise_image,
    _type_conversion,
    _crop_center,
    _compute_max_value,
    _compute_min_value,
    _log_normalisation,
    _zscale_normalisation,
    _conversiononly_normalisation,
    _asinh_normalisation,
    _expand,
)
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod


@pytest.fixture
def caplog(caplog):
    """Configure fitsbolt logger to capture logs with caplog"""
    # Ensure we're working with the logger instance
    fitsbolt_logger = logger._logger

    # Set up the logger to use the caplog handler
    original_handlers = fitsbolt_logger.handlers[:]
    original_level = fitsbolt_logger.level
    original_propagate = fitsbolt_logger.propagate

    # Clear existing handlers and add caplog handler
    fitsbolt_logger.handlers.clear()
    fitsbolt_logger.addHandler(caplog.handler)
    fitsbolt_logger.setLevel(logging.DEBUG)
    fitsbolt_logger.propagate = True

    yield caplog

    # Restore original configuration
    fitsbolt_logger.handlers.clear()
    fitsbolt_logger.handlers.extend(original_handlers)
    fitsbolt_logger.setLevel(original_level)
    fitsbolt_logger.propagate = original_propagate


def get_test_config(method=NormalisationMethod.CONVERSION_ONLY):
    """Returns a test config with the specified normalisation method"""
    cfg = create_config(
        size=[64, 64],
        normalisation_method=method,
        norm_maximum_value=None,
        norm_minimum_value=None,
        norm_crop_for_maximum_value=None,
        norm_log_calculate_minimum_value=False,
        norm_asinh_scale=[10.0, 10.0, 10.0],  # Default for ASINH
        norm_asinh_clip=[99.0, 99.0, 99.0],  # Default for ASINH
    )
    return cfg


def get_asinh_test_config(asinh_scale=[1.0, 1.0, 1.0], asinh_clip=[99.0, 99.0, 99.0]):
    """Create a test config specifically for ASINH normalisation"""
    cfg = get_test_config(NormalisationMethod.ASINH)
    if asinh_scale is not None:
        cfg.normalisation.asinh_scale = asinh_scale
    if asinh_clip is not None:
        cfg.normalisation.asinh_clip = asinh_clip
    return cfg


def create_gradient_rgb(height=16, width=16, dtype=np.uint8):
    """Create a test RGB image with gradients in different channels"""
    if dtype == np.uint16:
        max_val = 65535
    elif dtype == np.float32:
        max_val = 1e-3
    else:
        max_val = 255

    # Create gradients for each channel
    r = np.linspace(0, max_val, width)
    g = np.linspace(0, max_val / 2, width)
    b = np.linspace(max_val / 4, max_val, width)

    # Create meshgrids
    r_mesh, _ = np.meshgrid(r, np.linspace(0, max_val, height))
    g_mesh, _ = np.meshgrid(g, np.linspace(0, max_val / 2, height))
    b_mesh, _ = np.meshgrid(b, np.linspace(max_val / 4, max_val, height))

    # Stack channels
    image = np.stack([r_mesh, g_mesh, b_mesh], axis=2).astype(dtype)
    return image


def create_gradient_single_channel(height=16, width=16, dtype=np.uint8):
    """Create a test single channel image with gradient"""
    if dtype == np.uint16:
        max_val = 65535
    elif dtype == np.float32:
        max_val = 1e-3
    else:
        max_val = 255

    x = np.linspace(0, max_val, width)
    x_mesh, _ = np.meshgrid(x, np.linspace(0, max_val, height))
    return x_mesh.astype(dtype)


def create_multi_channel_image(height=16, width=16, dtype=np.uint8):
    """Create a test multi-channel image simulating 4 channels: V,Y,J,H astronomical bands
    with different intensity ranges to test proper scaling across channels"""
    if dtype == np.uint16:
        max_vals = [65535, 45000, 55000, 35000]  # Different max for each channel
    elif dtype == np.float32:
        max_vals = [1e-3, 7e-4, 8e-4, 5e-4]
    else:
        max_vals = [255, 180, 220, 180]

    channels = []
    for max_val in max_vals:
        # Create a different pattern for each channel
        x = np.linspace(0, max_val, width)
        y = np.linspace(0, max_val, height)
        x_mesh, y_mesh = np.meshgrid(x, y)
        # Add some variation per channel
        channel_data = (x_mesh + y_mesh) / 2
        channels.append(channel_data.astype(dtype))

    return np.stack(channels, axis=2)


class TestNormalisationMethod:
    """Test the NormalisationMethod enum."""

    def test_enum_values(self):
        """Test that the enum has expected values."""
        assert NormalisationMethod.CONVERSION_ONLY == 0
        assert NormalisationMethod.LOG == 1
        assert NormalisationMethod.ZSCALE == 2
        assert NormalisationMethod.ASINH == 3

    def test_get_options(self):
        """Test the get_options class method."""
        options = NormalisationMethod.get_options()
        expected = [
            ("ConversionOnly", 0),
            ("LogStretch", 1),
            ("ZscaleInterval", 2),
            ("Asinh", 3),
            ("Linear", 4),
            ("Midtones", 5),
        ]
        assert options == expected

    def test_get_test_methods(self):
        """Test the get_test_methods class method."""
        methods = NormalisationMethod.get_test_methods()
        expected = [
            NormalisationMethod.CONVERSION_ONLY,
            NormalisationMethod.LOG,
            NormalisationMethod.ZSCALE,
            NormalisationMethod.ASINH,
            NormalisationMethod.LINEAR,
            NormalisationMethod.MIDTONES,
        ]
        assert methods == expected


class TestNormalisationUtilities:
    """Test utility functions for normalisation."""

    def test_type_conversion(self):
        """Test _type_conversion function."""
        cfg = get_test_config()
        cfg.output_dtype = np.uint8

        # Test with float data
        float_data = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        converted = _type_conversion(float_data, cfg)
        assert converted.dtype == np.uint8
        assert converted.shape == float_data.shape

    def test_crop_center(self):
        """Test _crop_center function."""
        # Create a test image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[40:60, 40:60, :] = 255  # White square in center

        # Crop to 50x50
        cropped = _crop_center(image, 50, 50)
        assert cropped.shape == (50, 50, 3)

        # Test with crop larger than image
        large_crop = _crop_center(image, 150, 150)
        assert large_crop.shape == image.shape  # Should return original

    def test_compute_max_value(self):
        """Test _compute_max_value function."""
        cfg = get_test_config()

        # Test with regular image
        image = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        max_val = _compute_max_value(image, cfg)
        assert max_val == 6.0

        # Test with configured maximum value
        cfg.normalisation.maximum_value = 10.0
        max_val = _compute_max_value(image, cfg)
        assert max_val == 10.0

        # Test with crop for maximum value
        cfg.normalisation.maximum_value = None
        cfg.normalisation.crop_for_maximum_value = (1, 1)
        large_image = np.zeros((10, 10), dtype=np.float32)
        large_image[4:6, 4:6] = 100.0  # High values in center
        large_image[0, 0] = 200.0  # Even higher value outside center
        max_val = _compute_max_value(large_image, cfg)
        assert max_val == 100.0  # Should use center crop, not global max

    def test_compute_min_value(self):
        """Test _compute_min_value function."""
        cfg = get_test_config()

        # Test with regular image
        image = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        min_val = _compute_min_value(image, cfg)
        assert min_val == 1.0

        # Test with configured minimum value
        cfg.normalisation.minimum_value = -5.0
        min_val = _compute_min_value(image, cfg)
        assert min_val == -5.0

    def test_expand(self, caplog):
        """Test _expand function for asinh normalisation."""
        # Test with scalar
        result = _expand(0.5, 3)
        expected = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

        # Test with list
        result = _expand([1.0, 2.0, 3.0], 3)
        expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

        # Test with wrong length
        # Check logs for warnings
        caplog.clear()
        _expand([1.0, 2.0], 3)
        # Check that a warning was logged (format will depend on how the logger is configured)
        assert any(record.levelname == "WARNING" for record in caplog.records)
        caplog.clear()


class TestNormalisationMethods:
    """Test individual normalisation methods."""

    def test_conversion_only_uint8(self):
        """Test conversion-only normalisation with uint8 input."""
        cfg = get_test_config(NormalisationMethod.CONVERSION_ONLY)

        # Test with uint8 data (should pass through)
        image = np.array([[100, 150, 200]], dtype=np.uint8)
        result = _conversiononly_normalisation(image, cfg)
        np.testing.assert_array_equal(result, image)
        assert result.dtype == np.uint8

    def test_conversion_only_uint16(self):
        """Test conversion-only normalisation with uint16 input."""
        cfg = get_test_config(NormalisationMethod.CONVERSION_ONLY)

        # Test with uint16 data
        image = np.array([[0, 32767, 65535]], dtype=np.uint16)
        result = _conversiononly_normalisation(image, cfg)
        assert result.dtype == np.uint8
        # Should be approximately [0, 127, 255] after scaling
        expected = np.array([[0, 127, 255]], dtype=np.uint8)
        np.testing.assert_array_almost_equal(result, expected, decimal=0)

    def test_conversion_only_float32(self):
        """Test conversion-only normalisation with float32 input."""
        cfg = get_test_config(NormalisationMethod.CONVERSION_ONLY)

        # Test with float32 data
        image = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        result = _conversiononly_normalisation(image, cfg)
        assert result.dtype == np.uint8
        expected = np.array([[0, 127, 255]], dtype=np.uint8)
        np.testing.assert_array_almost_equal(result, expected, decimal=0)

    def test_conversion_only_edge_cases(self):
        """Test conversion-only normalisation with edge cases."""
        cfg = get_test_config(NormalisationMethod.CONVERSION_ONLY)

        # Test with invalid range (min >= max)
        image = np.array([[5.0, 5.0, 5.0]], dtype=np.float32)
        result = _conversiononly_normalisation(image, cfg)
        assert result.dtype == np.uint8
        # Should return zeros when min >= max
        assert np.all(result == 0)

    def test_log_normalisation(self):
        """Test log normalisation."""
        cfg = get_test_config(NormalisationMethod.LOG)

        # Test with positive data
        image = np.array([[1.0, 10.0, 100.0]], dtype=np.float32)
        result = _log_normalisation(image, cfg)
        assert result.dtype == np.uint8
        assert result.shape == image.shape

        # Test with calculate minimum value
        cfg.normalisation.log_calculate_minimum_value = True
        result = _log_normalisation(image, cfg)
        assert result.dtype == np.uint8

    def test_log_normalisation_edge_cases(self):
        """Test log normalisation with edge cases."""
        cfg = get_test_config(NormalisationMethod.LOG)

        # Test with minimum >= maximum
        image = np.array([[5.0, 4.0, 3.0]], dtype=np.float32)
        cfg.normalisation.minimum_value = 5.0
        cfg.normalisation.maximum_value = 3.0
        result = _log_normalisation(image, cfg)
        assert result.dtype == np.uint8
        # Should fall back to linear stretch

    def test_zscale_normalisation(self):
        """Test zscale normalisation."""
        cfg = get_test_config(NormalisationMethod.ZSCALE)

        # Test with varied data
        np.random.seed(42)
        image = np.random.normal(100, 20, (10, 10)).astype(np.float32)
        result = _zscale_normalisation(image, cfg)
        assert result.dtype == np.uint8
        assert result.shape == image.shape

    def test_zscale_normalisation_edge_case(self):
        """Test zscale normalisation with constant data."""
        cfg = get_test_config(NormalisationMethod.ZSCALE)

        # Test with constant data (should fall back to conversion only)
        image = np.full((5, 5), 100.0, dtype=np.float32)
        result = _zscale_normalisation(image, cfg)
        assert result.dtype == np.uint8

    def test_asinh_normalisation_single_channel(self):
        """Test asinh normalisation with single channel."""
        cfg = get_asinh_test_config(asinh_scale=[1.0], asinh_clip=[95.0])

        # Test with single channel data
        image = create_gradient_single_channel(dtype=np.float32)
        result = _asinh_normalisation(image, cfg)
        assert result.dtype == np.uint8
        assert result.shape == image.shape

    def test_asinh_normalisation_rgb(self):
        """Test asinh normalisation with RGB channels."""
        cfg = get_asinh_test_config()

        # Test with RGB data
        image = create_gradient_rgb(dtype=np.float32)
        result = _asinh_normalisation(image, cfg)
        assert result.dtype == np.uint8
        assert result.shape == image.shape

    def test_asinh_normalisation_edge_case(self):
        """Test asinh normalisation with edge cases."""
        cfg = get_asinh_test_config()

        # Test with constant data
        image = np.full((5, 5, 3), 100.0, dtype=np.float32)
        result = _asinh_normalisation(image, cfg)
        assert result.dtype == np.uint8
        # Should fall back to conversion only


class TestNormaliseImageIntegration:
    """Test the main normalise_image function."""

    def test_normalise_image_all_methods(self):
        """Test normalise_image with all normalisation methods."""
        methods = NormalisationMethod.get_test_methods()

        for method in methods:
            cfg = get_test_config(method)
            image = create_gradient_rgb(dtype=np.float32)

            result = _normalise_image(image, cfg)

            assert result.dtype == np.uint8, f"Failed for method {method}"
            assert result.shape == image.shape, f"Failed for method {method}"

    def test_normalise_image_single_channel(self):
        """Test normalise_image with single channel images."""
        methods = NormalisationMethod.get_test_methods()

        for method in methods:
            cfg = get_test_config(method)
            image = create_gradient_single_channel(dtype=np.float32)

            result = _normalise_image(image, cfg)

            assert result.dtype == np.uint8, f"Failed for method {method}"
            assert result.shape == image.shape, f"Failed for method {method}"

    def test_normalise_image_multi_channel(self):
        """Test normalise_image with multi-channel images."""
        cfg = get_asinh_test_config([1.0, 1.5, 2.0, 2.5])  # 4 channels
        image = create_multi_channel_image(dtype=np.float32)

        result = _normalise_image(image, cfg)

        assert result.dtype == np.uint8
        assert result.shape == image.shape

    def test_normalise_image_invalid_method(self):
        """Test normalise_image with invalid normalisation method."""
        cfg = get_test_config()
        cfg.normalisation_method = "invalid_method"  # Not a NormalisationMethod enum

        image = create_gradient_rgb(dtype=np.float32)
        result = _normalise_image(image, cfg)

        # Should fall back to conversion only
        assert result.dtype == np.uint8
        assert result.shape == image.shape

    def test_normalise_image_with_config_parameters(self):
        """Test normalise_image with various configuration parameters."""
        # Test with custom min/max values
        cfg = get_test_config(NormalisationMethod.CONVERSION_ONLY)
        cfg.normalisation.minimum_value = 0.0
        cfg.normalisation.maximum_value = 1000.0

        image = np.array([[500.0, 750.0, 1200.0]], dtype=np.float32)
        result = _normalise_image(image, cfg)

        assert result.dtype == np.uint8
        # Values should be clipped and scaled
        assert np.all(result <= 255)

    def test_normalise_image_crop_functionality(self):
        """Test normalise_image with crop for maximum value."""
        cfg = get_test_config(NormalisationMethod.CONVERSION_ONLY)
        cfg.normalisation.crop_for_maximum_value = (2, 2)

        # Create image with high values outside center
        image = np.zeros((10, 10), dtype=np.float32)
        image[4:6, 4:6] = 100.0  # Center values
        image[0, 0] = 1000.0  # High value outside center

        result = _normalise_image(image, cfg)
        assert result.dtype == np.uint8
        # Should use center region for normalisation, not global max


class TestNormalisationRobustness:
    """Test normalisation robustness with various data types and edge cases."""

    def test_different_dtypes(self):
        """Test normalisation with different input data types."""
        dtypes = [np.uint8, np.uint16, np.float32, np.float64]

        for dtype in dtypes:
            cfg = get_test_config(NormalisationMethod.CONVERSION_ONLY)

            if dtype == np.uint8:
                image = np.array([[0, 127, 255]], dtype=dtype)
            elif dtype == np.uint16:
                image = np.array([[0, 32767, 65535]], dtype=dtype)
            else:  # float types
                image = np.array([[0.0, 0.5, 1.0]], dtype=dtype)

            result = _normalise_image(image, cfg)
            assert result.dtype == np.uint8, f"Failed for dtype {dtype}"

    def test_extreme_values(self):
        """Test normalisation with extreme values."""
        cfg = get_test_config(NormalisationMethod.CONVERSION_ONLY)

        # Test with very large values
        image = np.array([[1e10, 2e10, 3e10]], dtype=np.float64)
        result = _normalise_image(image, cfg)
        assert result.dtype == np.uint8
        assert np.all(result >= 0) and np.all(result <= 255)

        # Test with very small values
        image = np.array([[1e-10, 2e-10, 3e-10]], dtype=np.float64)
        result = _normalise_image(image, cfg)
        assert result.dtype == np.uint8

    def test_nan_and_inf_handling(self):
        """Test normalisation with NaN and infinity values."""
        cfg = get_test_config(NormalisationMethod.CONVERSION_ONLY)

        # Test with NaN values
        image = np.array([[1.0, np.nan, 3.0]], dtype=np.float32)
        result = _normalise_image(image, cfg)
        assert result.dtype == np.uint8
        # Function should handle NaN gracefully

        # Test with infinity values
        image = np.array([[1.0, np.inf, 3.0]], dtype=np.float32)
        result = _normalise_image(image, cfg)
        assert result.dtype == np.uint8

    def test_empty_and_single_pixel(self):
        """Test normalisation with edge case image sizes."""
        cfg = get_test_config(NormalisationMethod.CONVERSION_ONLY)

        # Test with single pixel
        image = np.array([[100.0]], dtype=np.float32)
        result = _normalise_image(image, cfg)
        assert result.dtype == np.uint8
        assert result.shape == (1, 1)

        # Test with very small image
        image = np.array([[1.0, 2.0]], dtype=np.float32)
        result = _normalise_image(image, cfg)
        assert result.dtype == np.uint8
        assert result.shape == (1, 2)
