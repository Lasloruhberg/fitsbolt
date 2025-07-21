"""
Tests for the image IO utility functions in image_loader.py.
"""

import os
import numpy as np
import pytest
import shutil
import tempfile
from PIL import Image
from astropy.io import fits

from astro_loader.image_loader import (
    _read_image,
    _load_image,
    process_image,
    load_and_process_images,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from astro_loader.normalisation.NormalisationMethod import NormalisationMethod
from astro_loader.cfg.create_config import create_config


class TestImageIO:
    """Test class for image IO utilities."""

    @pytest.fixture
    def test_config(self):
        """Create test config for image loading."""
        cfg = create_config(
            size=[100, 100],  # Set a valid size for validation
            fits_extension=None,  # Default first extension
            normalisation_method=NormalisationMethod.CONVERSION_ONLY,
            norm_maximum_value=None,
            norm_minimum_value=None,
            norm_crop_for_maximum_value=None,
            norm_log_calculate_minimum_value=False,
        )
        return cfg

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

        # Create fully transparent test RGBA image
        transparent_img = np.zeros((100, 100, 4), dtype=np.uint8)
        transparent_img[25:75, 25:75, 1] = 255  # Green square
        transparent_img[25:75, 25:75, 3] = 0  # Fully transparent
        cls.transparent_path = os.path.join(cls.test_dir, "transparent.png")
        Image.fromarray(transparent_img).save(cls.transparent_path)

        # Create a complex RGBA image with varying alpha
        complex_rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        # Create a gradient pattern
        for i in range(100):
            for j in range(100):
                complex_rgba[i, j, 0] = min(255, i * 2)  # Red gradient
                complex_rgba[i, j, 1] = min(255, j * 2)  # Green gradient
                complex_rgba[i, j, 2] = min(255, (i + j))  # Blue gradient
                complex_rgba[i, j, 3] = min(255, (i + j) // 2 + 100)  # Alpha gradient
        cls.complex_rgba_path = os.path.join(cls.test_dir, "complex_rgba.png")
        Image.fromarray(complex_rgba).save(cls.complex_rgba_path)

        # Create a nested directory with an image
        nested_dir = os.path.join(cls.test_dir, "nested")
        os.makedirs(nested_dir)
        nested_img = np.zeros((50, 50, 3), dtype=np.uint8)
        nested_img[:, :, 1] = 200  # Green image
        cls.nested_path = os.path.join(nested_dir, "nested_image.jpg")
        Image.fromarray(nested_img).save(cls.nested_path)

        # Simple FITS file
        fits_data = np.zeros((100, 100), dtype=np.float32)
        fits_data[25:75, 25:75] = 1.0  # Bright square
        cls.fits_path = os.path.join(cls.test_dir, "test.fits")
        fits.writeto(cls.fits_path, fits_data, overwrite=True)

        # FITS file with multiple channels (RGB-like)
        multi_data = np.zeros((3, 100, 100), dtype=np.float32)
        multi_data[0, 25:75, 25:75] = 1.0  # Red
        multi_data[1, 35:85, 35:85] = 0.8  # Green
        multi_data[2, 45:95, 45:95] = 0.6  # Blue
        cls.multi_fits_path = os.path.join(cls.test_dir, "multi_channel.fits")
        fits.writeto(cls.multi_fits_path, multi_data, overwrite=True)

        # Create FITS with extreme values to test normalization
        extreme_data = np.zeros((100, 100), dtype=np.float32)
        extreme_data[10:40, 10:40] = -1000.0  # Very negative values
        extreme_data[50:80, 50:80] = 1000.0  # Very positive values
        cls.extreme_fits_path = os.path.join(cls.test_dir, "extreme_values.fits")
        fits.writeto(cls.extreme_fits_path, extreme_data, overwrite=True)

        # Keep track of all created image files
        cls.image_files = [
            cls.rgb_path,
            cls.gray_path,
            cls.rgba_path,
            cls.transparent_path,
            cls.complex_rgba_path,
            cls.nested_path,
            cls.fits_path,
            cls.multi_fits_path,
            cls.extreme_fits_path,
        ]

    @classmethod
    def teardown_class(cls):
        """Remove test files and directories."""
        try:
            shutil.rmtree(cls.test_dir)
        except (PermissionError, OSError) as e:
            # If we can't delete due to Windows file locking, just log it and continue
            print(f"Warning: Could not delete test directory: {e}")

    def test_supported_extensions(self):
        """Test that SUPPORTED_IMAGE_EXTENSIONS contains expected formats."""
        expected_extensions = {".fits", ".jpg", ".jpeg", ".png", ".tiff"}
        assert SUPPORTED_IMAGE_EXTENSIONS == expected_extensions

    def test_read_image_rgb(self, test_config):
        """Test reading an RGB image with _read_image."""
        img = _read_image(self.rgb_path, test_config)
        assert img.shape[2] == 3  # Should be RGB
        assert img.dtype in [np.uint8, np.float32, np.float64]  # PIL returns uint8

    def test_read_image_grayscale(self, test_config):
        """Test reading a grayscale image with _read_image."""
        img = _read_image(self.gray_path, test_config)
        # PIL might convert grayscale to RGB automatically
        assert img.ndim >= 2 and img.ndim <= 3

    def test_read_image_rgba(self, test_config):
        """Test reading an RGBA image with _read_image."""
        img = _read_image(self.rgba_path, test_config)
        assert img.shape[2] == 4  # Should preserve RGBA
        assert img.dtype in [np.uint8, np.float32, np.float64]

    def test_read_image_fits_simple(self, test_config):
        """Test reading a simple FITS file with _read_image."""
        img = _read_image(self.fits_path, test_config)
        assert img.ndim == 2  # Single channel FITS should be 2D
        assert np.issubdtype(img.dtype, np.floating), "Image data should be a floating-point type"

    def test_read_image_fits_multi_channel(self, test_config):
        """Test reading a multi-channel FITS file with _read_image."""
        img = _read_image(self.multi_fits_path, test_config)
        # Should handle multi-dimensional FITS appropriately
        assert img.ndim >= 2 and img.ndim <= 3

    def test_read_image_fits_extreme_values(self, test_config):
        """Test reading FITS with extreme values."""
        img = _read_image(self.extreme_fits_path, test_config)
        assert img.ndim == 2
        # Should handle extreme values without crashing

    def test_read_image_fits_extension_selection(self, test_config):
        """Test FITS extension selection functionality."""
        # Test with integer extension index
        test_config.fits_extension = 0
        img = _read_image(self.fits_path, test_config)
        assert img.ndim >= 2

        # Test invalid extension index
        test_config.fits_extension = 999
        with pytest.raises(IndexError):
            _read_image(self.fits_path, test_config)

    def test_read_image_unsupported_extension(self, test_config):
        """Test that unsupported file extensions raise AssertionError."""
        unsupported_path = os.path.join(self.test_dir, "test.bmp")
        # Create a fake file with unsupported extension
        with open(unsupported_path, "w") as f:
            f.write("fake file")

        with pytest.raises(AssertionError, match="Unsupported file extension"):
            _read_image(unsupported_path, test_config)

    def test_process_image_rgb_conversion(self, test_config):
        """Test process_image with RGB conversion."""
        # Test with grayscale input
        gray_data = np.zeros((50, 50), dtype=np.uint8)
        gray_data[10:40, 10:40] = 200

        processed = process_image(gray_data, test_config, convert_to_rgb=True)
        assert processed.shape == (
            test_config.size[0],
            test_config.size[1],
            3,
        )  # Should be converted to RGB
        assert processed.dtype == np.uint8

        # Test with RGBA input
        rgba_data = np.zeros((50, 50, 4), dtype=np.uint8)
        rgba_data[10:40, 10:40, :3] = [255, 128, 64]  # RGB values
        rgba_data[10:40, 10:40, 3] = 255  # Alpha

        processed = process_image(rgba_data, test_config, convert_to_rgb=True)
        assert processed.shape == (
            test_config.size[0],
            test_config.size[1],
            3,
        )  # Should be converted to RGB  # Should drop alpha channel
        assert processed.dtype == np.uint8

    def test_process_image_without_rgb_conversion(self, test_config):
        """Test process_image without RGB conversion."""
        gray_data = np.zeros((50, 50), dtype=np.uint8)
        gray_data[10:40, 10:40] = 200

        processed = process_image(gray_data, test_config, convert_to_rgb=False)
        assert len(processed.shape) == 2 or (len(processed.shape) == 3 and processed.shape[2] == 3)
        assert processed.dtype == np.uint8

    def test_process_image_with_resizing(self, test_config):
        """Test process_image with resizing."""
        test_config.size = (64, 64)

        rgb_data = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb_data[25:75, 25:75, 0] = 255  # Red square

        processed = process_image(rgb_data, test_config)
        assert processed.shape[:2] == (64, 64)
        assert processed.shape[2] == 3
        assert processed.dtype == np.uint8

    def test_load_image_integration(self, test_config):
        """Test _load_image function integration."""
        # Test with RGB image
        img = _load_image(self.rgb_path, test_config)
        assert img.shape[2] == 3  # Should be RGB
        assert img.dtype == np.uint8

        # Test with FITS image
        fits_img = _load_image(self.fits_path, test_config)
        assert fits_img.ndim >= 2
        assert fits_img.dtype == np.uint8

    def test_load_and_process_images_parallel(self, test_config):
        """Test load_and_process_images function with multiple images."""
        file_paths = [self.rgb_path, self.gray_path, self.rgba_path]

        results = load_and_process_images(
            file_paths, cfg=test_config, num_workers=2, show_progress=False
        )

        assert len(results) == 3  # All images should load successfully
        for filepath, image in results:
            assert filepath in file_paths
            assert image.shape[2] == 3  # All should be RGB
            assert image.dtype == np.uint8

    def test_load_and_process_images_with_config_params(self):
        """Test load_and_process_images with configuration parameters."""
        file_paths = [self.rgb_path, self.gray_path]

        results = load_and_process_images(
            file_paths,
            size=[64, 64],
            normalisation_method=NormalisationMethod.CONVERSION_ONLY,
            num_workers=1,
            show_progress=False,
        )

        assert len(results) == 2
        for filepath, image in results:
            assert image.shape[:2] == (64, 64)
            assert image.dtype == np.uint8

    def test_load_and_process_images_error_handling(self, test_config):
        """Test load_and_process_images with invalid file paths."""
        invalid_paths = ["/nonexistent/file.jpg", self.rgb_path]

        results = load_and_process_images(invalid_paths, cfg=test_config, show_progress=False)

        # Should only return results for valid files
        assert len(results) == 1
        assert results[0][0] == self.rgb_path

    def test_rgba_to_rgb_conversion_values(self, test_config):
        """Test that RGBA to RGB conversion handles alpha channel correctly."""
        # Create a test RGBA image with varying alpha and patterns
        width, height = 100, 100
        test_rgba = np.zeros((height, width, 4), dtype=np.uint8)

        # Create a pattern where:
        # - Upper left quadrant (red with full alpha)
        # - Upper right quadrant (green with half alpha)
        # - Lower left quadrant (blue with quarter alpha)
        # - Lower right quadrant (white with zero alpha)
        test_rgba[: height // 2, : width // 2, 0] = 255  # Red
        test_rgba[: height // 2, : width // 2, 3] = 255  # Full alpha

        test_rgba[: height // 2, width // 2 :, 1] = 255  # Green
        test_rgba[: height // 2, width // 2 :, 3] = 128  # Half alpha

        test_rgba[height // 2 :, : width // 2, 2] = 255  # Blue
        test_rgba[height // 2 :, : width // 2, 3] = 64  # Quarter alpha

        test_rgba[height // 2 :, width // 2 :, 0:3] = 255  # White
        test_rgba[height // 2 :, width // 2 :, 3] = 0  # Zero alpha

        # Test RGB conversion
        rgb_img = process_image(test_rgba, test_config, convert_to_rgb=True)

        # Test shape and type
        assert rgb_img.shape == (height, width, 3), "RGBA should convert to RGB shape"
        assert rgb_img.dtype == np.uint8, "RGBA conversion should maintain uint8 type"

        # Test that colors are preserved correctly (alpha is dropped, not blended)
        # Colors with full alpha should be preserved exactly
        assert np.all(
            rgb_img[: height // 2, : width // 2, 0] == 255
        ), "Red with full alpha should be preserved"
        assert np.all(
            rgb_img[: height // 2, : width // 2, 1] == 0
        ), "Red channel should have no green"

    def test_fits_loading_parallel_with_extension_int(self, test_config):
        """Test parallel loading of FITS files with extension index 0."""
        test_config.fits_extension = 0
        file_paths = [self.fits_path, self.multi_fits_path]

        results = load_and_process_images(
            file_paths, cfg=test_config, num_workers=2, show_progress=False
        )

        assert len(results) == 2
        for filepath, image in results:
            assert filepath in file_paths
            assert image.shape[2] == 3  # Should be converted to RGB
            assert image.dtype == np.uint8

    def test_fits_extension_primary(self, test_config):
        """Test reading FITS file with 'PRIMARY' extension name."""
        # Set the extension to 'PRIMARY' which should be equivalent to index 0
        test_config.fits_extension = "PRIMARY"

        try:
            img = _read_image(self.fits_path, test_config)
            assert img.ndim >= 2
            assert np.issubdtype(img.dtype, np.floating)
        except KeyError:
            # Some FITS files might not have named extensions
            pytest.skip("FITS file does not have a 'PRIMARY' named extension")

    def test_greyscale_fits_to_rgb_mapping(self, test_config):
        """Test conversion of greyscale FITS data to RGB."""
        # Load a single-channel FITS file
        test_config.fits_extension = 0

        # First get the raw grayscale data
        gray_fits = _read_image(self.fits_path, test_config)
        assert gray_fits.ndim == 2, "Test requires a 2D (grayscale) FITS image"

        # Now process with RGB conversion
        rgb_fits = process_image(gray_fits, test_config, convert_to_rgb=True)

        # Verify RGB conversion
        assert rgb_fits.shape[2] == 3, "Grayscale FITS should be converted to RGB"
        assert rgb_fits.dtype == np.uint8, "Processed image should be uint8"

        # All RGB channels should be equal for a grayscale source
        r_channel = rgb_fits[:, :, 0]
        g_channel = rgb_fits[:, :, 1]
        b_channel = rgb_fits[:, :, 2]

        # Channels should be identical or very similar
        assert np.allclose(r_channel, g_channel, rtol=1e-5, atol=1)
        assert np.allclose(r_channel, b_channel, rtol=1e-5, atol=1)

    def test_fits_extension_list_truncation(self, test_config):
        """Test that extension list [0,1,2,3] drops the last channel when converting to RGB."""
        # Create a FITS file with 4 extensions for this test
        # First, create 4-channel test data with HDUs that have actual extensions
        primary_hdu = fits.PrimaryHDU(np.zeros((50, 50), dtype=np.float32))
        hdu_list = fits.HDUList([primary_hdu])

        # Add 3 image extensions with different data
        for i in range(3):
            ext_data = np.zeros((50, 50), dtype=np.float32)
            ext_data[10:40, 10:40] = i + 1  # Each extension has different values
            ext_hdu = fits.ImageHDU(ext_data)
            ext_hdu.header["EXTNAME"] = f"EXT{i + 1}"
            hdu_list.append(ext_hdu)

        # Create a temporary FITS file with multiple extensions
        temp_fits_path = os.path.join(self.test_dir, "four_channel.fits")
        hdu_list.writeto(temp_fits_path, overwrite=True)

        # Test 1: Using all extensions by index
        test_config.fits_extension = [0, 1, 2, 3]

        # This should fail as extension 3 doesn't exist (we have 0,1,2)
        with pytest.raises(IndexError, match="out of bounds"):
            _read_image(temp_fits_path, test_config)

        # Try with the extensions that actually exist
        test_config.fits_extension = [0, 1, 2]
        img = _read_image(temp_fits_path, test_config)

        # The shape should be (50, 50, 3) after reading
        assert img.shape == (50, 50, 3), "Should read all 3 channels"

        # Now process with RGB conversion
        processed = process_image(img, test_config, convert_to_rgb=True)

        # After processing, it should still be (50, 50, 3) as we have exactly 3 channels
        assert processed.shape == (
            50,
            50,
            3,
        ), "Should maintain 3 channels during RGB conversion"

        # Test 2: Now create a file with exactly 4 extensions to test truncation
        # Add one more extension
        ext_data = np.zeros((50, 50), dtype=np.float32)
        ext_data[10:40, 10:40] = 4  # Fourth extension
        ext_hdu = fits.ImageHDU(ext_data)
        ext_hdu.header["EXTNAME"] = "EXT4"
        hdu_list.append(ext_hdu)

        # Write the updated file with 4 extensions
        temp_fits_path2 = os.path.join(self.test_dir, "four_ext_channel.fits")
        hdu_list.writeto(temp_fits_path2, overwrite=True)

        # Set config to use all 4 extensions
        test_config.fits_extension = [0, 1, 2, 3]

        # This should work now
        img_four = _read_image(temp_fits_path2, test_config)

        # The shape should be (50, 50, 4) after reading
        assert img_four.shape == (50, 50, 4), "Should read all 4 channels"

        # Now process with RGB conversion
        processed_four = process_image(img_four, test_config, convert_to_rgb=True)

        # After processing, it should be (50, 50, 3) - dropping the last channel
        assert processed_four.shape == (
            50,
            50,
            3,
        ), "Should drop the 4th channel during RGB conversion"

        # Check that only the first 3 channels were kept
        for i in range(3):
            expected_val = i + 1  # Based on our test data pattern
            assert np.any(
                np.isclose(processed_four[:, :, i], expected_val, rtol=1e-5, atol=1)
            ), f"Channel {i} should have data"

        # Test 3: Try with extension names instead of indices
        test_config.fits_extension = ["PRIMARY", "EXT1", "EXT2"]
        img_named = _read_image(temp_fits_path2, test_config)
        assert img_named.shape == (50, 50, 3), "Should read named extensions"

        # Test error case - non-existent extension name
        test_config.fits_extension = ["PRIMARY", "EXT1", "NONEXISTENT"]
        with pytest.raises(KeyError, match="not found"):
            _read_image(temp_fits_path2, test_config)

    def test_fits_extension_list_with_two_extensions(self, test_config):
        """Test handling of a list with two extensions and error cases."""
        # Create a FITS file with named extensions
        primary_hdu = fits.PrimaryHDU(np.zeros((50, 50), dtype=np.float32))
        primary_hdu.header["EXTNAME"] = "PRIMARY"

        ext1_data = np.zeros((50, 50), dtype=np.float32)
        ext1_data[10:40, 10:40] = 1.0  # First extension
        ext1_hdu = fits.ImageHDU(ext1_data)
        ext1_hdu.header["EXTNAME"] = "RED"

        ext2_data = np.zeros((50, 50), dtype=np.float32)
        ext2_data[20:30, 20:30] = 2.0  # Second extension
        ext2_hdu = fits.ImageHDU(ext2_data)
        ext2_hdu.header["EXTNAME"] = "GREEN"

        # Create a temporary FITS file with two extensions
        hdu_list = fits.HDUList([primary_hdu, ext1_hdu, ext2_hdu])
        temp_fits_path = os.path.join(self.test_dir, "two_ext_channel.fits")
        hdu_list.writeto(temp_fits_path, overwrite=True)

        try:
            # Test 1: Using two extensions by index
            test_config.fits_extension = [0, 1]
            img = _read_image(temp_fits_path, test_config)

            # Image should have shape (50, 50, 2) after reading
            assert img.shape == (50, 50, 2), "Should read both channels into shape (H, W, 2)"

            # Process with RGB conversion - should work by filling in a zero channel
            processed = process_image(img, test_config, convert_to_rgb=True)

            # Check that we have 3 channels now
            assert processed.shape == (50, 50, 3), "Should have 3 channels after RGB conversion"

            # The third channel should be all zeros or very close to zero
            assert np.allclose(
                processed[:, :, 2], 0, atol=1
            ), "Third channel should be zero or close to zero"

            # Test 2: Using extension names
            test_config.fits_extension = ["PRIMARY", "RED"]
            img_named = _read_image(temp_fits_path, test_config)
            assert img_named.shape == (50, 50, 2), "Should read named extensions"

            # Test 3: Test error with non-existent extension
            test_config.fits_extension = [0, 99]  # Index 99 doesn't exist
            with pytest.raises(IndexError, match="out of bounds"):
                _read_image(temp_fits_path, test_config)

            # Test 4: Test error with non-existent named extension
            test_config.fits_extension = ["PRIMARY", "NONEXISTENT"]
            with pytest.raises(KeyError, match="not found"):
                _read_image(temp_fits_path, test_config)

            # Test 5: Test with extensions that have different shapes (should fail)
            # Create a new HDU with different shape
            odd_shape_data = np.zeros((40, 60), dtype=np.float32)  # Different shape
            odd_shape_hdu = fits.ImageHDU(odd_shape_data)
            odd_shape_hdu.header["EXTNAME"] = "ODD_SHAPE"

            # Add to a new HDU list
            odd_hdu_list = fits.HDUList([primary_hdu, ext1_hdu, odd_shape_hdu])
            odd_fits_path = os.path.join(self.test_dir, "odd_shape.fits")
            odd_hdu_list.writeto(odd_fits_path, overwrite=True)

            # Try to load extensions with different shapes
            test_config.fits_extension = [0, 1, 2]
            with pytest.raises(ValueError, match="different shapes"):
                _read_image(odd_fits_path, test_config)

        finally:
            # Clean up temporary files
            for path in [temp_fits_path, odd_fits_path if "odd_fits_path" in locals() else None]:
                if path is not None and os.path.exists(path):
                    try:
                        os.remove(path)
                    except (OSError, PermissionError):
                        pass
