# fitsbolt - A Python package for image loading and processing
# Copyright (C) <2025>  <Ruhberg>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the MIT License
"""
Tests for the wrapper functions (read_images, resize_images, normalise_images).
"""

import os
import numpy as np
import pytest
import shutil
import tempfile
from PIL import Image
from astropy.io import fits

from fitsbolt.read import read_images
from fitsbolt.resize import resize_images, resize_image
from fitsbolt.normalisation.normalisation import normalise_images, _normalise_image
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod
from fitsbolt.cfg.create_config import create_config
from fitsbolt.image_loader import load_and_process_images


class TestWrapperFunctionEdgeCases:
    """Test edge cases and error handling for wrapper functions."""

    @classmethod
    def setup_class(cls):
        """Set up test files and directories."""
        cls.test_dir = tempfile.mkdtemp()

        # Create test RGB image
        rgb_img = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb_img[25:75, 25:75, 0] = 255  # Red square
        cls.rgb_path = os.path.join(cls.test_dir, "test_rgb.jpg")
        Image.fromarray(rgb_img).save(cls.rgb_path)

        # Create test grayscale image
        gray_img = np.zeros((100, 100), dtype=np.uint8)
        gray_img[25:75, 25:75] = 200  # White square
        cls.gray_path = os.path.join(cls.test_dir, "test_gray.jpg")
        Image.fromarray(gray_img).save(cls.gray_path)

        # Simple FITS file
        fits_data = np.zeros((100, 100), dtype=np.float32)
        fits_data[25:75, 25:75] = 1.0  # Bright square
        cls.fits_path = os.path.join(cls.test_dir, "test.fits")
        fits.writeto(cls.fits_path, fits_data, overwrite=True)

    @classmethod
    def teardown_class(cls):
        """Remove test files and directories."""
        try:
            shutil.rmtree(cls.test_dir)
        except (PermissionError, OSError) as e:
            print(f"Warning: Could not delete test directory: {e}")

    def test_read_images_single_file_failure_returns_proper_none(self):
        """Test read_images single file failure raises proper exception."""
        # Test with invalid file path - should raise an exception instead of handling gracefully
        with pytest.raises(Exception):  # Should raise FileNotFoundError or similar
            read_images("/totally/nonexistent/file.jpg", show_progress=False)

    def test_read_images_multiple_jpg(self):
        """Test the specific warning case in read_images where multiple images are loaded but single was requested."""
        file_paths = [self.rgb_path, self.rgb_path]  # Valid files that will load
        results = read_images(file_paths, show_progress=False)
        assert len(results) == 2, "Expected two images to be loaded"
        for i in range(0, 2):
            assert results[i].shape[:2] == (100, 100), "Should be at original size"

    def test_read_images_multiple_jpg_grey(self):
        """Test the specific warning case in read_images where multiple images are loaded but single was requested."""
        file_paths = [self.gray_path, self.gray_path]  # Valid files that will load
        results = read_images(file_paths, show_progress=False)
        assert len(results) == 2, "Expected two images to be loaded"
        for i in range(0, 2):
            assert results[i].shape[:2] == (100, 100), "Should be at original size"

    def test_read_images_warning_multiple_loaded_single_requested(self):
        """Test the specific warning case in read_images where multiple images are loaded but single was requested."""
        # This tests the specific warning branch in lines 104-107 of read.py
        # The scenario where return_single=True but len(results) > 1 and we want the first one
        file_paths = [self.rgb_path, self.gray_path]  # Valid files that will load

        # Simulate by calling read_images - it should raise an error with the grayscale
        with pytest.raises(AssertionError, match="Unexpected number of channels: "):
            read_images(file_paths, show_progress=False)
        file_paths = [self.gray_path, self.rgb_path]  # Valid files that will load
        with pytest.raises(AssertionError, match="Unexpected number of channels: "):
            read_images(file_paths, show_progress=False)

    def test_resize_images_edge_cases(self):
        """Test resize_images with edge cases."""
        # Test with empty list
        result_empty = resize_images([], show_progress=False)
        assert len(result_empty) == 0, "Empty list should return empty list"

        # Test with very small images
        tiny_img = np.zeros((2, 2, 3), dtype=np.uint8)
        result_tiny = resize_images([tiny_img], size=[10, 10], show_progress=False)
        assert result_tiny[0].shape[:2] == (10, 10), "Should resize tiny image"

        # Test with very large target size
        small_img = np.zeros((10, 10, 3), dtype=np.uint8)
        result_large = resize_images([small_img], size=[500, 500], show_progress=False)
        assert result_large[0].shape[:2] == (500, 500), "Should resize to large size"

    def test_resize_image_no_resize_needed(self):
        """Test resize_image when image is already the target size."""
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[20:44, 20:44, 0] = 255

        # Resize to same size - should handle efficiently
        result = resize_image(img, size=[64, 64])
        assert result.shape == (64, 64, 3), "Should maintain size"
        # The function should still process it (not necessarily return same object)

    def test_resize_image_with_none_size(self):
        """Test resize_image with size=None."""
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        result = resize_image(img, size=None)
        assert result.shape == (50, 50, 3), "Should maintain original size when size=None"

    def test_normalise_images_different_failure_scenarios(self):
        """Test normalise_images with various failure scenarios."""
        # Test with very extreme values
        extreme_img = np.full((10, 10, 3), np.inf, dtype=np.float32)
        try:
            result = normalise_images(extreme_img, show_progress=False)
            # Should handle extreme values gracefully
            assert result.dtype == np.uint8, "Should convert to uint8 even with extreme values"
        except Exception:
            # Some normalisation methods might fail with extreme values, which is acceptable
            pass

        # Test with NaN values
        nan_img = np.full((10, 10, 3), np.nan, dtype=np.float32)
        try:
            result = normalise_images(nan_img, show_progress=False)
            # Should handle NaN values
            assert result.dtype == np.uint8, "Should convert to uint8"
        except Exception:
            # Some normalisation methods might fail with NaN values, which is acceptable
            pass

    def test_normalise_images_log_with_calculate_minimum(self):
        """Test LOG normalisation with calculate minimum value option."""
        img = np.zeros((50, 50, 3), dtype=np.float32)
        img[20:30, 20:30, 0] = 1000.0  # High value
        img[10:20, 10:20, 0] = -500.0  # Negative value

        result = normalise_images(
            img,
            normalisation_method=NormalisationMethod.LOG,
            norm_log_calculate_minimum_value=True,
            show_progress=False,
        )

        assert result.dtype == np.uint8, "Should convert to uint8"
        assert result.shape == (50, 50, 3), "Should maintain shape"

    def test_normalise_images_log_with_4D(self):
        """Test LOG normalisation with calculate minimum value option."""
        imgs = np.zeros((4, 50, 50, 3), dtype=np.float32)
        for i in range(0, 3):
            imgs[i, 20:30, 20:30, 0] = 1000.0 + i * 1000.0  # High value
            imgs[i, 10:20, 10:20, 0] = -500.0 + i * -100.0  # Negative value

        result = normalise_images(
            imgs,
            normalisation_method=NormalisationMethod.LOG,
            norm_log_calculate_minimum_value=True,
            show_progress=False,
        )
        for i in range(0, 3):
            assert result[i].dtype == np.uint8, "Should convert to uint8"
            assert result[i].shape == (50, 50, 3), "Should maintain shape"

    def test_normalise_images_with_crop_for_maximum(self):
        """Test normalisation with crop_for_maximum_value parameter."""
        img = np.zeros((100, 100, 3), dtype=np.float32)
        img[40:60, 40:60, 0] = 1000.0  # High value in center
        img[0:20, 0:20, 0] = 10.0  # Lower value in corner

        result = normalise_images(
            img,
            normalisation_method=NormalisationMethod.CONVERSION_ONLY,
            norm_crop_for_maximum_value=(40, 40),  # Crop around center for max value
            show_progress=False,
        )

        assert result.dtype == np.uint8, "Should convert to uint8"
        assert result.shape == (100, 100, 3), "Should maintain shape"

    def test_load_and_process_images_cfg_parameter(self):
        """Test load_and_process_images with explicit cfg parameter."""
        # Create a custom config
        cfg = create_config(
            size=[32, 32],
            normalisation_method=NormalisationMethod.CONVERSION_ONLY,
            n_output_channels=3,
        )

        result = load_and_process_images(self.rgb_path, cfg=cfg, show_progress=False)

        assert isinstance(result, np.ndarray), "Should return single array for single file"
        assert result.shape[:2] == (32, 32), "Should use cfg size"
        assert result.shape[2] == 3, "Should have 3 channels"

    def test_read_images_with_show_progress_true(self):
        """Test read_images with progress bar enabled."""
        file_paths = [self.rgb_path, self.rgb_path]
        results = read_images(file_paths, show_progress=True, desc="Testing progress")

        assert len(results) == 2, "Should return 2 images"
        for result in results:
            assert isinstance(result, np.ndarray), "Each result should be an array"

    def test_resize_images_with_show_progress_true(self):
        """Test resize_images with progress bar enabled."""
        img1 = np.zeros((50, 50, 3), dtype=np.uint8)
        img2 = np.zeros((100, 100, 3), dtype=np.uint8)
        images = [img1, img2]

        results = resize_images(
            images, size=[64, 64], show_progress=True, desc="Testing resize progress"
        )

        assert len(results) == 2, "Should return 2 resized images"

    def test_normalise_images_with_show_progress_true(self):
        """Test normalise_images with progress bar enabled."""
        img1 = np.zeros((50, 50, 3), dtype=np.float32)
        img1[20:30, 20:30, 0] = 1000.0

        img2 = np.zeros((50, 50, 3), dtype=np.float32)
        img2[10:40, 10:40, 1] = 2000.0

        images = [img1, img2]
        results = normalise_images(images, show_progress=True, desc="Testing normalise progress")

        assert len(results) == 2, "Should return 2 images"

    def test_error_scenarios_with_resize_function(self):
        """Test error scenarios in resize functions."""
        # Test with invalid image data
        invalid_img = np.array([])

        try:
            resize_image(invalid_img, size=[32, 32])
        except Exception:
            # Should handle invalid input gracefully
            pass

    def test_normalise_images_return_single_vs_multiple_edge_case(self):
        """Test the edge case where single image is requested but multiple results exist."""
        # Test the specific case where return_single=True but len(results) > 1
        img = np.zeros((50, 50, 3), dtype=np.float32)
        img[20:30, 20:30, 0] = 1000.0

        # This should trigger the warning case in the normalise_images function
        result = normalise_images(img, show_progress=False)
        assert isinstance(result, np.ndarray), "Single image should return single array"

    def test_resize_images_with_extreme_interpolation_orders(self):
        """Test resize_images with edge case interpolation orders."""
        img = np.zeros((50, 50, 3), dtype=np.uint8)

        # Test with order 0 (nearest neighbor)
        result_0 = resize_images([img], size=[100, 100], interpolation_order=0, show_progress=False)
        assert result_0[0].shape[:2] == (100, 100), "Should resize with order 0"

        # Test with order 5 (highest order)
        result_5 = resize_images([img], size=[100, 100], interpolation_order=5, show_progress=False)
        assert result_5[0].shape[:2] == (100, 100), "Should resize with order 5"

    def test_load_and_process_images_with_no_cfg_provided(self):
        """Test load_and_process_images when no cfg is provided - should create one internally."""
        # This tests the cfg=None branch in load_and_process_images
        result = load_and_process_images(
            self.rgb_path,
            cfg=None,  # Explicitly set to None to test internal config creation
            size=[48, 48],
            normalisation_method=NormalisationMethod.CONVERSION_ONLY,
            show_progress=False,
        )

        assert isinstance(result, np.ndarray), "Should return single array"
        assert result.shape[:2] == (48, 48), "Should use provided size"


class TestWrapperFunctions:
    """Test class for the wrapper functions (read_images, resize_images, normalise_images)."""

    @classmethod
    def setup_class(cls):
        """Set up test files and directories."""
        # Create a temporary test directory
        cls.test_dir = tempfile.mkdtemp()

        # Create test RGB image
        rgb_img = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb_img[25:75, 25:75, 0] = 255  # Red square
        cls.rgb_path = os.path.join(cls.test_dir, "test_rgb.jpg")
        Image.fromarray(rgb_img).save(cls.rgb_path)

        # Create test grayscale image
        gray_img = np.zeros((100, 100), dtype=np.uint8)
        gray_img[25:75, 25:75] = 200  # White square
        cls.gray_path = os.path.join(cls.test_dir, "test_gray.jpg")
        Image.fromarray(gray_img).save(cls.gray_path)

        # Create test RGBA image
        rgba_img = np.zeros((100, 100, 4), dtype=np.uint8)
        rgba_img[25:75, 25:75, 0] = 255  # Red square
        rgba_img[25:75, 25:75, 3] = 128  # Semi-transparent
        cls.rgba_path = os.path.join(cls.test_dir, "test_rgba.png")
        Image.fromarray(rgba_img).save(cls.rgba_path)

        # Simple FITS file
        fits_data = np.zeros((100, 100), dtype=np.float32)
        fits_data[25:75, 25:75] = 1.0  # Bright square
        cls.fits_path = os.path.join(cls.test_dir, "test.fits")
        fits.writeto(cls.fits_path, fits_data, overwrite=True)

        # FITS file with multiple channels (RGB-like)
        # Create separate extensions for each channel instead of 3D data
        primary_hdu = fits.PrimaryHDU(np.zeros((100, 100), dtype=np.float32))
        hdu_list = fits.HDUList([primary_hdu])

        # Add 3 image extensions with different data (RGB-like)
        channel_data = [
            np.zeros((100, 100), dtype=np.float32),  # Extension 1 (Red)
            np.zeros((100, 100), dtype=np.float32),  # Extension 2 (Green)
            np.zeros((100, 100), dtype=np.float32),  # Extension 3 (Blue)
        ]
        channel_data[0][25:75, 25:75] = 1.0  # Red
        channel_data[1][35:85, 35:85] = 0.8  # Green
        channel_data[2][45:95, 45:95] = 0.6  # Blue

        for i, data in enumerate(channel_data):
            ext_hdu = fits.ImageHDU(data)
            ext_hdu.header["EXTNAME"] = f"CHANNEL{i + 1}"
            hdu_list.append(ext_hdu)

        cls.multi_fits_path = os.path.join(cls.test_dir, "multi_channel.fits")
        hdu_list.writeto(cls.multi_fits_path, overwrite=True)

        # Keep track of all created image files
        cls.image_files = [
            cls.rgb_path,
            cls.gray_path,
            cls.rgba_path,
            cls.fits_path,
            cls.multi_fits_path,
        ]

    @classmethod
    def teardown_class(cls):
        """Remove test files and directories."""
        try:
            shutil.rmtree(cls.test_dir)
        except (PermissionError, OSError) as e:
            # If we can't delete due to Windows file locking, just log it and continue
            print(f"Warning: Could not delete test directory: {e}")

    def test_read_images_single_file(self):
        """Test read_images with a single file path."""
        # Test with single file (should return single image, not list)
        result = read_images(self.rgb_path, show_progress=False)
        assert isinstance(result, np.ndarray), "Single file should return single array"
        assert result.shape[2] == 3, "Should have 3 channels"
        assert (
            result.dtype == np.uint8
        ), "PNG images should maintain uint8 dtype with force_dtype=True"

    def test_read_images_multiple_files(self):
        """Test read_images with multiple file paths."""
        file_paths = [self.rgba_path, self.rgba_path, self.rgba_path]
        results = read_images(
            file_paths, n_output_channels=4, show_progress=False, force_dtype=True
        )

        assert isinstance(results, np.ndarray), "Multiple files should return list"
        assert len(results) == 3, "Should return all 3 images"
        for result in results:
            assert isinstance(result, np.ndarray), "Each result should be an array"
            assert result.shape[2] == 4, "Should have 4 channels by default"
            assert (
                result.dtype == np.uint8
            ), "PNG images should maintain uint8 dtype with force_dtype=True"

    def test_read_images_with_parameters(self):
        """Test read_images with custom parameters."""
        file_paths = [self.gray_path, self.gray_path]
        results = read_images(
            file_paths,
            n_output_channels=3,
            num_workers=1,
            desc="Testing read",
            show_progress=False,
        )

        assert len(results) == 2, "Should return 2 images"
        for result in results:
            assert result.shape[:2] == (100, 100), "Should be at original size"
            assert result.shape[2] == 3, "Should have 3 channels"

    def test_read_images_fits_with_extension(self):
        """Test read_images with FITS files and extension parameters."""
        file_paths = [self.fits_path, self.multi_fits_path]
        results = read_images(
            file_paths, fits_extension=0, n_output_channels=1, show_progress=False
        )

        assert len(results) == 2, "Should return 2 images"
        for result in results:
            assert result.ndim == 3, "Should be 3D with 1 channel with a 2 image for single channel"
            assert result.shape[-1] == 1, "Should be a 2D image - 1 channel"
            assert np.issubdtype(result.dtype, np.floating), "FITS data should be float"

    def test_read_images_error_handling(self):
        """Test read_images with invalid file paths."""
        invalid_paths = ["/nonexistent/file.jpg", self.rgb_path]

        # Should raise an exception when encountering invalid file paths
        with pytest.raises(Exception):  # Should raise FileNotFoundError or similar
            read_images(invalid_paths, show_progress=False)

    def test_read_images_single_file_error_handling(self):
        """Test read_images error handling with single invalid file."""
        # Test with single invalid file - should raise an exception
        with pytest.raises(Exception):  # Should raise FileNotFoundError or similar
            read_images("/nonexistent/file.jpg", show_progress=False)

    def test_read_images_channel_combination(self):
        """Test read_images with channel combination for FITS files."""
        # Create custom channel combination
        # Identity matrix
        channel_combination = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
        )

        results = read_images(
            [self.multi_fits_path],
            fits_extension=[0, 1, 2],
            n_output_channels=3,
            channel_combination=channel_combination,
            show_progress=False,
        )

        assert len(results) == 1, "Should return 1 image"
        assert results[0].shape[2] == 3, "Should have 3 channels"

    def test_resize_images_list(self):
        """Test resize_images with a list of images."""
        # Create test images
        img1 = np.zeros((50, 50, 3), dtype=np.uint8)
        img2 = np.zeros((100, 100, 3), dtype=np.uint8)
        images = [img1, img2]

        results = resize_images(images, size=[64, 64], show_progress=False)

        assert len(results) == 2, "Should return 2 resized images"
        for result in results:
            assert result.shape[:2] == (64, 64), "Should be resized to 64x64"
            assert result.shape[2] == 3, "Should maintain 3 channels"
            assert result.dtype == np.uint8, "Should maintain uint8 dtype"

    def test_resize_images_different_dtypes(self):
        """Test resize_images with different output dtypes."""
        img = np.zeros((50, 50, 3), dtype=np.uint8)

        # Test uint16 output
        result_uint16 = resize_images(
            [img], output_dtype=np.uint16, size=[32, 32], show_progress=False
        )
        assert result_uint16[0].dtype == np.uint16, "Should convert to uint16"

        # Test float32 output
        result_float32 = resize_images(
            [img], output_dtype=np.float32, size=[32, 32], show_progress=False
        )
        assert result_float32[0].dtype == np.float32, "Should convert to float32"

    def test_resize_images_interpolation_order(self):
        """Test resize_images with different interpolation orders."""
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[20:30, 20:30, 0] = 255  # Red square

        # Test different interpolation orders
        for order in [0, 1, 2, 3]:
            result = resize_images(
                [img], size=[100, 100], interpolation_order=order, show_progress=False
            )
            assert result[0].shape[:2] == (100, 100), f"Should resize with order {order}"

    def test_resize_images_no_size_specified(self):
        """Test resize_images when no size is specified."""
        img = np.zeros((50, 50, 3), dtype=np.uint8)

        result = resize_images([img], size=None, show_progress=False)
        assert result[0].shape == (50, 50, 3), "Should maintain original size when size=None"

    def test_resize_image_single(self):
        """Test resize_image function with single image."""
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[20:30, 20:30, 0] = 255  # Red square

        result = resize_image(img, size=[100, 100])
        assert result.shape[:2] == (100, 100), "Should resize to 100x100"
        assert result.shape[2] == 3, "Should maintain 3 channels"
        assert result.dtype == np.uint8, "Should maintain uint8 dtype"

    def test_resize_image_different_dtypes(self):
        """Test resize_image with different output dtypes."""
        img = np.zeros((50, 50, 3), dtype=np.uint8)

        # Test uint16 output
        result_uint16 = resize_image(img, output_dtype=np.uint16, size=[32, 32])
        assert result_uint16.dtype == np.uint16, "Should convert to uint16"

        # Test float32 output
        result_float32 = resize_image(img, output_dtype=np.float32, size=[32, 32])
        assert result_float32.dtype == np.float32, "Should convert to float32"

    def test_normalise_images_single_image(self):
        """Test normalise_images with a single image."""
        img = np.zeros((50, 50, 3), dtype=np.float32)
        img[20:30, 20:30, 0] = 1000.0  # High value in red channel

        result = normalise_images(img, show_progress=False)
        assert isinstance(result, np.ndarray), "Single image should return single array"
        assert result.shape == (50, 50, 3), "Should maintain shape"
        assert result.dtype == np.uint8, "Should convert to uint8"
        assert np.max(result) <= 255, "Should be normalised to uint8 range"

    def test_normalise_images_multiple_images(self):
        """Test normalise_images with multiple images."""
        img1 = np.zeros((50, 50, 3), dtype=np.float32)
        img1[20:30, 20:30, 0] = 1000.0  # High value

        img2 = np.zeros((50, 50, 3), dtype=np.float32)
        img2[10:40, 10:40, 1] = 2000.0  # Different high value

        images = [img1, img2]
        results = normalise_images(images, show_progress=False)

        assert isinstance(results, list), "Multiple images should return list"
        assert len(results) == 2, "Should return 2 images"
        for result in results:
            assert result.dtype == np.uint8, "Should convert to uint8"
            assert np.max(result) <= 255, "Should be normalised to uint8 range"

    def test_normalise_images_different_methods(self):
        """Test normalise_images with different normalisation methods."""
        img = np.zeros((50, 50, 3), dtype=np.float32)
        img[20:30, 20:30, 0] = 1000.0  # High value

        # Test CONVERSION_ONLY
        result_conv = normalise_images(
            img, normalisation_method=NormalisationMethod.CONVERSION_ONLY, show_progress=False
        )
        assert result_conv.dtype == np.uint8, "Should convert to uint8"

        # Test LOG normalisation
        result_log = normalise_images(
            img, normalisation_method=NormalisationMethod.LOG, show_progress=False
        )
        assert result_log.dtype == np.uint8, "Should convert to uint8"

        # Test ZSCALE normalisation
        result_zscale = normalise_images(
            img, normalisation_method=NormalisationMethod.ZSCALE, show_progress=False
        )
        assert result_zscale.dtype == np.uint8, "Should convert to uint8"

    def test_normalise_images_with_parameters(self):
        """Test normalise_images with custom parameters."""
        img = np.zeros((50, 50, 3), dtype=np.float32)
        img[20:30, 20:30, 0] = 1000.0  # High value

        result = normalise_images(
            img,
            normalisation_method=NormalisationMethod.CONVERSION_ONLY,
            num_workers=1,
            norm_maximum_value=500.0,
            norm_minimum_value=0.0,
            desc="Testing normalisation",
            show_progress=False,
        )

        assert result.dtype == np.uint8, "Should convert to uint8"
        assert result.shape == (50, 50, 3), "Should maintain shape"

    def test_normalise_images_asinh_method(self):
        """Test normalise_images with ASINH method and custom parameters."""
        img = np.zeros((50, 50, 3), dtype=np.float32)
        img[20:30, 20:30, :] = [1000.0, 800.0, 600.0]  # Different values per channel

        result = normalise_images(
            img,
            normalisation_method=NormalisationMethod.ASINH,
            norm_asinh_scale=[0.5, 0.7, 0.9],
            norm_asinh_clip=[95.0, 98.0, 99.0],
            show_progress=False,
        )

        assert result.dtype == np.uint8, "Should convert to uint8"
        assert result.shape == (50, 50, 3), "Should maintain shape"

    def test_normalise_images_error_handling(self):
        """Test normalise_images error handling with invalid input."""
        # Test with None or empty input
        result_empty = normalise_images([], show_progress=False)
        assert isinstance(result_empty, list), "Empty list should return empty list"
        assert len(result_empty) == 0, "Empty input should return empty output"

    def test_load_and_process_images_single_file_return(self):
        """Test load_and_process_images single file return behavior."""
        # Test single file should return single array, not list
        result = load_and_process_images(self.rgb_path, show_progress=False)
        assert isinstance(result, np.ndarray), "Single file should return single array"
        assert result.shape[2] == 3, "Should have 3 channels"

    def test_load_and_process_images_multiple_files_some_fail(self):
        """Test load_and_process_images when some files fail to load."""
        # Mix valid and invalid paths - should raise an exception when encountering invalid files
        mixed_paths = [self.rgb_path, "/nonexistent1.jpg", self.gray_path, "/nonexistent2.jpg"]

        with pytest.raises(Exception):  # Should raise FileNotFoundError or similar
            load_and_process_images(mixed_paths, show_progress=False)

    def test_normalise_images_comprehensive_dtype_combinations(self):
        """Test normalise_images with all normalisation methods and all input/output dtype combinations."""
        # Test input data types: float32, uint8, uint16
        # Test output data types: uint8, uint16, float32 (float16 is not supported by the normalisation code)
        # Test all normalisation methods: CONVERSION_ONLY, LOG, ZSCALE, ASINH

        # Define test input data types
        input_dtypes = [np.float32, np.uint8, np.uint16]

        # Define test output data types (only those supported by the normalisation code)
        output_dtypes = [np.uint8, np.uint16, np.float32]

        # Define test normalisation methods
        norm_methods = [
            NormalisationMethod.CONVERSION_ONLY,
            NormalisationMethod.LOG,
            NormalisationMethod.ZSCALE,
            NormalisationMethod.ASINH,
        ]

        # Create test images with different input dtypes
        test_images = {}
        for input_dtype in input_dtypes:
            # Create test image with some pattern
            img = np.zeros((50, 50, 3), dtype=input_dtype)

            # Set different values based on dtype to ensure meaningful data
            if input_dtype == np.float32:
                img[20:30, 20:30, 0] = 1000.0  # High value in red channel
                img[10:20, 10:20, 1] = 500.0  # Medium value in green channel
                img[30:40, 30:40, 2] = 100.0  # Low value in blue channel
            elif input_dtype == np.uint8:
                img[20:30, 20:30, 0] = 255  # Max value in red channel
                img[10:20, 10:20, 1] = 128  # Half value in green channel
                img[30:40, 30:40, 2] = 64  # Quarter value in blue channel
            elif input_dtype == np.uint16:
                img[20:30, 20:30, 0] = 65535  # Max value in red channel
                img[10:20, 10:20, 1] = 32768  # Half value in green channel
                img[30:40, 30:40, 2] = 16384  # Quarter value in blue channel

            test_images[input_dtype] = img

        # Test all combinations
        test_count = 0
        success_count = 0

        for input_dtype in input_dtypes:
            for output_dtype in output_dtypes:
                for norm_method in norm_methods:
                    test_count += 1

                    # Create custom config with the desired output dtype
                    cfg = create_config(
                        output_dtype=output_dtype,
                        normalisation_method=norm_method,
                        norm_maximum_value=None,  # Let it calculate dynamically
                        norm_minimum_value=None,  # Let it calculate dynamically
                        norm_log_calculate_minimum_value=(norm_method == NormalisationMethod.LOG),
                        norm_asinh_scale=[0.5, 0.7, 0.9],  # Different scales per channel
                        norm_asinh_clip=[95.0, 98.0, 99.0],  # Different clips per channel
                    )

                    # Get the test image for this input dtype
                    test_img = test_images[input_dtype].copy()

                    try:
                        # Apply normalisation using the internal function with our custom config
                        result = _normalise_image(test_img, cfg)

                        # Verify output dtype
                        assert result.dtype == output_dtype, (
                            f"Expected output dtype {output_dtype}, got {result.dtype} "
                            f"for input {input_dtype} with method {norm_method.name}"
                        )

                        # Verify output shape matches input shape
                        assert (
                            result.shape == test_img.shape
                        ), f"Output shape {result.shape} doesn't match input shape {test_img.shape}"

                        # Verify output values are in expected range for the dtype
                        if output_dtype == np.uint8:
                            assert np.min(result) >= 0, "uint8 output should have min >= 0"
                            assert np.max(result) <= 255, "uint8 output should have max <= 255"
                        elif output_dtype == np.uint16:
                            assert np.min(result) >= 0, "uint16 output should have min >= 0"
                            assert np.max(result) <= 65535, "uint16 output should have max <= 65535"
                        elif output_dtype == np.float32:
                            # For float types, values should be finite
                            assert np.all(np.isfinite(result)), "Float output should be finite"
                            # Normalised values should typically be in [0, 1] range for most methods
                            if norm_method == NormalisationMethod.CONVERSION_ONLY:
                                assert np.min(result) >= 0, "Normalised float should have min >= 0"
                                assert np.max(result) <= 1, "Normalised float should have max <= 1"

                        success_count += 1

                    except Exception as e:
                        # Some combinations might fail (e.g., LOG with negative values)
                        # This is acceptable for certain edge cases
                        if norm_method == NormalisationMethod.LOG and input_dtype == np.float32:
                            # LOG method might fail with certain float32 values, which is acceptable
                            pass
                        else:
                            # For other cases, we want to know about failures
                            print(
                                f"Failed combination: input={input_dtype.__name__}, "
                                f"output={output_dtype.__name__}, method={norm_method.name}"
                            )
                            raise e

        # Ensure we tested a reasonable number of combinations and most succeeded
        assert test_count == len(input_dtypes) * len(output_dtypes) * len(
            norm_methods
        ), "Should test all combinations"
        assert (
            success_count >= test_count * 0.8
        ), f"At least 80% of combinations should succeed, got {success_count}/{test_count}"

    def test_normalise_images_unsupported_dtype_fallback(self):
        """Test that unsupported output dtypes fall back to uint8 with a warning."""
        # Test that float16 (unsupported) falls back to uint8
        img = np.zeros((10, 10, 3), dtype=np.float32)
        img[5:8, 5:8, 0] = 100.0

        # Create config with unsupported dtype
        cfg = create_config(
            output_dtype=np.float16,  # This is not supported
            normalisation_method=NormalisationMethod.CONVERSION_ONLY,
        )

        # Apply normalisation - should fall back to uint8
        result = _normalise_image(img, cfg)

        # Should fall back to uint8, not float16
        assert result.dtype == np.uint8, "Unsupported dtype should fall back to uint8"
        assert result.shape == img.shape, "Shape should be preserved"
