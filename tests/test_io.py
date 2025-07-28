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

from fitsbolt.image_loader import (
    _read_image,
    _load_image,
    _process_image,
    load_and_process_images,
)
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod
from fitsbolt.cfg.create_config import create_config, SUPPORTED_IMAGE_EXTENSIONS


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

    def recreate_config(self, base_config=None, **kwargs):
        """Helper function to recreate config with modified parameters.

        This ensures that dependent parameters like channel_combination are
        properly recalculated when core parameters like n_output_channels change.

        Args:
            base_config: Base configuration to use as template (optional)
            **kwargs: Parameters to override in the new configuration

        Returns:
            New configuration object with updated parameters
        """
        # Default config parameters
        config_params = {
            "size": [100, 100],
            "fits_extension": None,
            "normalisation_method": NormalisationMethod.CONVERSION_ONLY,
            "norm_maximum_value": None,
            "norm_minimum_value": None,
            "norm_crop_for_maximum_value": None,
            "norm_log_calculate_minimum_value": False,
            "n_output_channels": 3,  # Explicitly set default
        }

        # If base_config is provided, extract its parameters
        if base_config is not None:
            for key in config_params.keys():
                if hasattr(base_config, key):
                    config_params[key] = getattr(base_config, key)

        # Override with any provided kwargs
        config_params.update(kwargs)

        # Create new config with updated parameters
        return create_config(**config_params)

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
        # Use recreate_config to ensure n_output_channels is properly set for RGB conversion
        cfg = self.recreate_config(test_config, n_output_channels=3)
        img = _read_image(self.gray_path, cfg)
        # Should convert grayscale to RGB format
        assert img.ndim == 3 and img.shape[2] == 3

    def test_read_image_rgba(self, test_config):
        """Test reading an RGBA image with _read_image."""
        # Use recreate_config to ensure n_output_channels is properly set for RGBA
        cfg = self.recreate_config(test_config, n_output_channels=4)
        img = _read_image(self.rgba_path, cfg)
        assert img.shape[2] == 4  # Should preserve RGBA
        assert img.dtype in [np.uint8, np.float32, np.float64]

    def test_read_image_fits_simple(self, test_config):
        """Test reading a simple FITS file with _read_image."""
        # Use recreate_config to ensure n_output_channels is properly set for single channel
        cfg = self.recreate_config(test_config, n_output_channels=1)
        img = _read_image(self.fits_path, cfg)
        assert img.ndim == 2  # Single channel FITS should be 2D when n_output_channels=1
        assert np.issubdtype(img.dtype, np.floating), "Image data should be a floating-point type"

    def test_read_image_fits_multi_channel(self, test_config):
        """Test reading a multi-channel FITS file with _read_image."""
        img = _read_image(self.multi_fits_path, test_config)
        # Should handle multi-dimensional FITS appropriately
        assert img.ndim >= 2 and img.ndim <= 3

    def test_read_image_fits_extreme_values(self, test_config):
        """Test reading FITS with extreme values."""
        # Use recreate_config to ensure n_output_channels is properly set for single channel
        cfg = self.recreate_config(test_config, n_output_channels=1)
        img = _read_image(self.extreme_fits_path, cfg)
        assert img.ndim == 2  # Single channel FITS should be 2D when n_output_channels=1
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
        """Test process_image preserves channel structure."""
        # Test with RGB input (should remain RGB)
        rgb_data = np.zeros((50, 50, 3), dtype=np.uint8)
        rgb_data[10:40, 10:40, 0] = 200  # Red channel

        processed = _process_image(
            rgb_data,
            test_config,
        )
        assert processed.shape == (
            test_config.size[0],
            test_config.size[1],
            3,
        )  # Should preserve RGB
        assert processed.dtype == np.uint8

        # Test with RGBA input (should remain RGBA - no channel conversion in _process_image)
        rgba_data = np.zeros((50, 50, 4), dtype=np.uint8)
        rgba_data[10:40, 10:40, :3] = [255, 128, 64]  # RGB values
        rgba_data[10:40, 10:40, 3] = 255  # Alpha

        processed = _process_image(
            rgba_data,
            test_config,
        )
        assert processed.shape == (
            test_config.size[0],
            test_config.size[1],
            4,
        )  # Should preserve RGBA - no channel conversion in _process_image
        assert processed.dtype == np.uint8

    def test_process_image_without_rgb_conversion(self, test_config):
        """Test process_image with single channel output."""
        gray_data = np.zeros((50, 50), dtype=np.uint8)
        gray_data[10:40, 10:40] = 200

        # Create config for single channel output to avoid RGB conversion
        cfg = self.recreate_config(test_config, n_output_channels=1)
        processed = _process_image(gray_data, cfg)
        assert (
            len(processed.shape) == 2
        ), f"Expected 2D output for single channel, got shape {processed.shape}"
        assert processed.dtype == np.uint8

    def test_process_image_with_resizing(self, test_config):
        """Test process_image with resizing."""
        # Use recreate_config to properly set the size
        cfg = self.recreate_config(test_config, size=(64, 64))

        rgb_data = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb_data[25:75, 25:75, 0] = 255  # Red square

        processed = _process_image(rgb_data, cfg)
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
        for image in results:
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
        for image in results:
            assert image.shape[:2] == (64, 64)
            assert image.dtype == np.uint8

    def test_load_and_process_images_error_handling(self, test_config):
        """Test load_and_process_images with invalid file paths."""
        invalid_paths = ["/nonexistent/file.jpg", self.rgb_path]

        results = load_and_process_images(invalid_paths, cfg=test_config, show_progress=False)

        # Should only return results for valid files
        assert len(results) == 1
        # The one valid result should be an image array
        assert isinstance(results[0], np.ndarray)
        assert results[0].shape[2] == 3  # Should be RGB

    def test_rgba_to_rgb_conversion_values(self, test_config):
        """Test that RGBA to RGB conversion handles alpha channel correctly."""
        # Create config with explicit n_output_channels=3 for RGB conversion
        cfg = self.recreate_config(test_config, n_output_channels=3)

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

        test_rgba_path = os.path.join(self.test_dir, "test_rgba.png")
        Image.fromarray(test_rgba).save(test_rgba_path)

        # Test RGB conversion using _read_image with RGBA file
        cfg = self.recreate_config(test_config, n_output_channels=3)
        rgb_img = _read_image(test_rgba_path, cfg)

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

    def test_channel_conversion_in_read_image(self, test_config):
        """Test that _read_image properly converts channels based on n_output_channels."""
        # Test RGBA to RGB conversion using _read_image
        cfg_rgb = self.recreate_config(test_config, n_output_channels=3)
        rgb_img = _read_image(self.rgba_path, cfg_rgb)
        assert rgb_img.shape[2] == 3, "RGBA should be converted to RGB with n_output_channels=3"

        # Test RGBA preservation using _read_image
        cfg_rgba = self.recreate_config(test_config, n_output_channels=4)
        rgba_img = _read_image(self.rgba_path, cfg_rgba)
        assert rgba_img.shape[2] == 4, "RGBA should be preserved with n_output_channels=4"

        # Test grayscale to RGB conversion using _read_image
        cfg_gray_to_rgb = self.recreate_config(test_config, n_output_channels=3)
        gray_to_rgb_img = _read_image(self.gray_path, cfg_gray_to_rgb)
        assert (
            gray_to_rgb_img.shape[2] == 3
        ), "Grayscale should be converted to RGB with n_output_channels=3"

    def test_fits_loading_parallel_with_extension_int(self, test_config):
        """Test parallel loading of FITS files with extension index 0."""
        # Use recreate_config to properly set fits_extension
        cfg = self.recreate_config(test_config, fits_extension=0)
        file_paths = [self.fits_path, self.multi_fits_path]

        results = load_and_process_images(file_paths, cfg=cfg, num_workers=2, show_progress=False)

        assert len(results) == 2
        for image in results:
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
        cfg = self.recreate_config(test_config, n_output_channels=1)
        # First get the raw grayscale data
        gray_fits = _read_image(self.fits_path, cfg)
        assert gray_fits.ndim == 2, "Test requires a 2D (grayscale) FITS image"

        # check rgb mapping
        cfg = self.recreate_config(test_config, n_output_channels=3)
        # First get the raw grayscale data
        rgb_fits = _read_image(self.fits_path, cfg)
        assert rgb_fits.ndim == 3, "mapping should convert to 3D RGB image"
        assert rgb_fits.shape[2] == 3, "Grayscale FITS should have three channels after mapping"

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
        for i in range(4):
            ext_data = np.zeros((50, 50), dtype=np.float32)
            ext_data[10:40, 10:40] = i + 1  # Each extension has different values
            ext_hdu = fits.ImageHDU(ext_data)
            ext_hdu.header["EXTNAME"] = f"EXT{i + 1}"
            hdu_list.append(ext_hdu)

        # Create a temporary FITS file with multiple extensions
        temp_fits_path = os.path.join(self.test_dir, "four_channel.fits")
        hdu_list.writeto(temp_fits_path, overwrite=True)

        # Test 1: Using all extensions by index
        test_config.fits_extension = [0, 1, 2, 3, 4]
        test_config.n_output_channels = 3  # Explicitly set output channels to 3
        # Setup channel_combination array - identity matrix for first 3 channels
        channel_comb = np.zeros((3, 5))  # 3 output channels x 4 input extensions
        for i in range(3):  # Only map the first 3 channels
            channel_comb[i, i] = 1
        channel_comb[2, 3] = 1  # map the 4th extension also to the 3rd output channel
        channel_comb[2, 4] = 1  # map the 5th extension also to the 3rd output channel
        test_config.channel_combination = channel_comb

        # This should fail as extension 3 doesn't exist (we have 0,1,2)
        # We're trying to access four_channel.fits with indices [0,1,2,3] but testing with multi_channel.fits
        with pytest.raises(IndexError, match="out of bounds"):
            _read_image(os.path.join(self.test_dir, "multi_channel.fits"), test_config)

        # Try with the extensions that actually exist
        test_config.fits_extension = [0, 1, 2]
        test_config.n_output_channels = 3  # Explicitly set output channels to 3
        # Setup channel_combination array - identity matrix
        channel_comb = np.zeros((3, 3))  # 3 output channels x 3 input extensions
        for i in range(3):
            channel_comb[i, i] = 1.0  # Use floating point to ensure no integer division issues
        test_config.channel_combination = channel_comb

        img = _read_image(temp_fits_path, test_config)

        # The shape should be (50, 50, 3) after reading
        assert img.shape == (50, 50, 3), "Should read all 3 channels"

        # Now process with RGB conversion
        processed = _process_image(
            img,
            test_config,
        )

        # After processing, the shape should match n_output_channels
        assert processed.shape == (
            test_config.size[0],
            test_config.size[1],
            3,
        ), "Should match n_output_channels=3 after RGB conversion"

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
        test_config.n_output_channels = 3  # Explicitly set output channels to 3
        # Setup channel_combination array for first 3 channels
        channel_comb = np.zeros((3, 4))  # 3 output channels x 4 input extensions
        for i in range(3):  # Only map the first 3 channels
            channel_comb[i, i] = 1
        test_config.channel_combination = channel_comb

        # This should work now
        img_four = _read_image(temp_fits_path2, test_config)

        # The shape should be (50, 50, 3) after reading since n_output_channels=3
        assert img_four.shape == (
            50,
            50,
            test_config.n_output_channels,
        ), f"Should read data with shape matching n_output_channels={test_config.n_output_channels}, got {img_four.shape}"

        # Now process with RGB conversion
        processed_four = _process_image(
            img_four,
            test_config,
        )

        # After processing, it should be (50, 50, 3)
        expected_shape = (test_config.size[0], test_config.size[1], test_config.n_output_channels)
        assert processed_four.shape == (
            expected_shape
        ), f"Final shape should be ({expected_shape}) matching n_output_channels, got {processed_four.shape}"

        # Check that we have data in at least some of the channels
        assert np.any(processed_four > 0), "At least one channel should have data"

        # Test 3: Try with extension names instead of indices
        test_config.fits_extension = ["PRIMARY", "EXT1", "EXT2"]
        test_config.n_output_channels = 3  # Explicitly set output channels to 3
        # Setup channel_combination array - identity matrix
        channel_comb = np.zeros((3, 3))  # 3 output channels x 3 input extensions
        for i in range(3):
            channel_comb[i, i] = 1
        test_config.channel_combination = channel_comb

        img_named = _read_image(temp_fits_path2, test_config)
        assert img_named.shape == (50, 50, 3), "Should read named extensions"

        # Test error case - non-existent extension name
        test_config.fits_extension = ["PRIMARY", "EXT1", "NONEXISTENT"]
        test_config.n_output_channels = 3  # Explicitly set output channels to 3
        # Setup channel_combination array - identity matrix
        channel_comb = np.zeros((3, 3))  # 3 output channels x 3 input extensions
        for i in range(3):
            channel_comb[i, i] = 1
        test_config.channel_combination = channel_comb

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
            test_config.n_output_channels = 3  # Explicitly set output channels to 3
            # Create channel_combination matrix for 2 channels to 3 output channels
            channel_comb = np.zeros((3, 2))  # 3 output channels x 2 input extensions
            channel_comb[0, 0] = 1  # Map first extension to R
            channel_comb[1, 1] = 1  # Map second extension to G
            channel_comb[2, 0] = (
                0.5  # Map first extension to B with half intensity (so there is no 0 sum)
            )
            # B channel will remain zeros
            test_config.channel_combination = channel_comb

            img = _read_image(temp_fits_path, test_config)

            # Image should have shape (50, 50, 3) after reading since n_output_channels=3
            assert img.shape == (50, 50, 3), "Should have n_output_channels=3 dimensions"

            # Process with RGB conversion - should work by filling in a zero channel
            processed = _process_image(
                img,
                test_config,
            )

            # Check that we have 3 channels now
            assert processed.shape == (
                test_config.size[0],
                test_config.size[1],
                test_config.n_output_channels,
            ), f"Should have {test_config.n_output_channels} channels after RGB conversion, got {processed.shape}"

            # At least one channel should have non-zero data
            assert np.any(processed[:, :, 0] > 0) or np.any(
                processed[:, :, 1] > 0
            ), "At least one channel should have data"

            # Test 2: Using extension names
            test_config.fits_extension = ["PRIMARY", "RED"]
            test_config.n_output_channels = 3  # Explicitly set output channels to 3
            # Same channel_combination as before
            channel_comb = np.zeros((3, 2))  # 3 output channels x 2 input extensions
            channel_comb[0, 0] = 1  # Map PRIMARY to R
            channel_comb[1, 1] = 1  # Map RED to G
            test_config.channel_combination = channel_comb

            img_named = _read_image(temp_fits_path, test_config)
            assert img_named.shape == (
                50,
                50,
                3,
            ), "Should have n_output_channels=3 dimensions for named extensions"

            # Test 3: Test error with non-existent extension
            test_config.fits_extension = [0, 99]  # Index 99 doesn't exist
            test_config.n_output_channels = 3  # Explicitly set output channels to 3
            # Setup channel_combination array
            channel_comb = np.zeros((3, 2))  # 3 output channels x 2 input extensions
            channel_comb[0, 0] = 1  # Map first extension to R
            channel_comb[1, 1] = 1  # Map second extension to G
            test_config.channel_combination = channel_comb

            with pytest.raises(IndexError, match="out of bounds"):
                _read_image(temp_fits_path, test_config)

            # Test 4: Test error with non-existent named extension
            test_config.fits_extension = ["PRIMARY", "NONEXISTENT"]
            test_config.n_output_channels = 3  # Explicitly set output channels to 3
            # Setup channel_combination array
            channel_comb = np.zeros((3, 2))  # 3 output channels x 2 input extensions
            channel_comb[0, 0] = 1  # Map first extension to R
            channel_comb[1, 1] = 1  # Map second extension to G
            test_config.channel_combination = channel_comb

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
            test_config.n_output_channels = 3  # Explicitly set output channels to 3
            # Setup channel_combination array
            channel_comb = np.zeros((3, 3))  # 3 output channels x 3 input extensions
            for i in range(3):
                channel_comb[i, i] = 1
            test_config.channel_combination = channel_comb

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

    def test_automatic_channel_combination_creation(self):
        """Test automatic creation of channel_combination array with different fits_extension configurations."""
        # Test 1: Single fits_extension with n_output_channels=3
        cfg_single = create_config(
            size=[100, 100],
            fits_extension=[0],  # Single extension as a list
            n_output_channels=3,  # RGB output
            normalisation_method=NormalisationMethod.CONVERSION_ONLY,
        )

        # Channel combination should be created as a 3x1 matrix with all elements set to 1
        assert hasattr(
            cfg_single, "channel_combination"
        ), "channel_combination should be automatically created"
        assert isinstance(cfg_single.channel_combination, np.ndarray)
        assert cfg_single.channel_combination.shape == (
            3,
            1,
        ), "Should be a 3x1 matrix for single extension"
        assert np.all(
            cfg_single.channel_combination == 1
        ), "All elements should be 1 for single extension"

        # Test 2: Multiple fits_extension with matching n_output_channels=3
        cfg_three = create_config(
            size=[100, 100],
            fits_extension=[0, 1, 2],  # Three extensions
            n_output_channels=3,  # RGB output
            normalisation_method=NormalisationMethod.CONVERSION_ONLY,
        )

        # Channel combination should be created as a 3x3 identity matrix
        assert hasattr(
            cfg_three, "channel_combination"
        ), "channel_combination should be automatically created"
        assert isinstance(cfg_three.channel_combination, np.ndarray)
        assert cfg_three.channel_combination.shape == (
            3,
            3,
        ), "Should be a 3x3 matrix for three extensions"
        # Should be identity matrix
        for i in range(3):
            for j in range(3):
                if i == j:
                    assert (
                        cfg_three.channel_combination[i, j] == 1
                    ), f"Diagonal element ({i},{j}) should be 1"
                else:
                    assert (
                        cfg_three.channel_combination[i, j] == 0
                    ), f"Non-diagonal element ({i},{j}) should be 0"

        # Test 3: Test with custom channel_combination provided
        custom_combination = np.array(
            [[0.5, 0.5, 0], [0, 1, 0], [0, 0, 1]]  # R = 0.5*ext1 + 0.5*ext2  # G = ext2  # B = ext3
        )

        cfg_custom = create_config(
            size=[100, 100],
            fits_extension=[0, 1, 2],
            n_output_channels=3,
            normalisation_method=NormalisationMethod.CONVERSION_ONLY,
            channel_combination=custom_combination,
        )

        # The custom channel_combination should be used as-is
        assert hasattr(cfg_custom, "channel_combination"), "channel_combination should exist"
        assert np.array_equal(
            cfg_custom.channel_combination, custom_combination
        ), "Custom combination should be used"

        # Test 4: Test with fits_extension length != n_output_channels (should fail)
        with pytest.raises(ValueError, match="Length of fits_extensions does not match"):
            create_config(
                size=[100, 100],
                fits_extension=[0, 1, 2, 3, 4],  # Five extensions
                n_output_channels=3,  # RGB output
                normalisation_method=NormalisationMethod.CONVERSION_ONLY,
            )

        # Test 5: Create a FITS file with exactly 4 extensions for real-world testing
        primary_hdu = fits.PrimaryHDU(np.zeros((50, 50), dtype=np.float32))
        hdu_list = fits.HDUList([primary_hdu])

        # Add image extensions with different data
        for i in range(3):  # Add 3 more extensions
            ext_data = np.zeros((50, 50), dtype=np.float32)
            ext_data[10:40, 10:40] = i + 1  # Each extension has different values
            ext_hdu = fits.ImageHDU(ext_data)
            ext_hdu.header["EXTNAME"] = f"EXT{i + 1}"
            hdu_list.append(ext_hdu)

        # Create a temporary FITS file
        temp_fits_path = os.path.join(self.test_dir, "test_channel_combination.fits")
        hdu_list.writeto(temp_fits_path, overwrite=True)

        try:
            # Test with explicit n_output_channels and fits_extension
            test_cfg = create_config(
                size=[50, 50],
                fits_extension=[0, 1, 2],
                n_output_channels=3,
                normalisation_method=NormalisationMethod.CONVERSION_ONLY,
            )

            # Read and process the image
            img = _read_image(temp_fits_path, test_cfg)
            assert img.shape == (50, 50, 3), "Should read all 3 channels"

            processed = _process_image(
                img,
                test_cfg,
            )
            assert processed.shape == (50, 50, 3), "Should maintain 3 channels"

            # Verify that at least one channel has data (channel 1 should have data from EXT1)
            assert np.any(processed[:, :, 1] > 0), "Channel 1 should have data"

        finally:
            # Clean up temporary file
            if os.path.exists(temp_fits_path):
                try:
                    os.remove(temp_fits_path)
                except (OSError, PermissionError):
                    pass

    def test_channel_combination_weighted_averaging(self, test_config):
        """Test that weights in channel_combination are properly applied when averaging channels."""
        # Create a FITS file with 2 extensions having distinct values
        primary_hdu = fits.PrimaryHDU(np.zeros((50, 50), dtype=np.float32))

        # First extension - uniformly filled with value 2.0
        ext1_data = np.ones((50, 50), dtype=np.float32) * 2.0
        ext1_hdu = fits.ImageHDU(ext1_data)
        ext1_hdu.header["EXTNAME"] = "EXT1"

        # Second extension - uniformly filled with value 4.0
        ext2_data = np.ones((50, 50), dtype=np.float32) * 4.0
        ext2_hdu = fits.ImageHDU(ext2_data)
        ext2_hdu.header["EXTNAME"] = "EXT2"

        # Create a temporary FITS file with these extensions
        hdu_list = fits.HDUList([primary_hdu, ext1_hdu, ext2_hdu])
        temp_fits_path = os.path.join(self.test_dir, "weighted_avg_test.fits")
        hdu_list.writeto(temp_fits_path, overwrite=True)

        try:
            # Configure to use all 3 extensions with a specific channel combination
            test_config.fits_extension = [0, 1, 2]
            test_config.n_output_channels = 3

            # Channel combination with weighted averaging for the third channel:
            # Channel 0 (output) = Channel 0 (input/PRIMARY)
            # Channel 1 (output) = Channel 1 (input/EXT1) - value 2.0
            # Channel 2 (output) = 0.25*Channel 1 (input/EXT1) + 0.75*Channel 2 (input/EXT2)
            #                    = 0.25*2.0 + 0.75*4.0 = 0.5 + 3.0 = 3.5
            channel_comb = np.zeros((3, 3))
            channel_comb[0, 0] = 1.0  # Map PRIMARY to output channel 0
            channel_comb[1, 1] = 1.0  # Map EXT1 to output channel 1
            channel_comb[2, 1] = 0.25  # Map 25% of EXT1 to output channel 2
            channel_comb[2, 2] = 0.75  # Map 75% of EXT2 to output channel 2
            test_config.channel_combination = channel_comb

            # Read the image
            img = _read_image(temp_fits_path, test_config)

            # Check output shape
            assert img.shape == (50, 50, 3), "Should create an image with 3 channels"

            # Check values:
            # Channel 0 should be zeros (from PRIMARY)
            assert np.allclose(
                img[:, :, 0], 0.0, atol=1e-5
            ), "Channel 0 should be zeros from PRIMARY"

            # Channel 1 should be 2.0 (from EXT1)
            assert np.allclose(img[:, :, 1], 2.0, atol=1e-5), "Channel 1 should be 2.0 from EXT1"

            # Channel 2 should be the weighted average: 0.25*2.0 + 0.75*4.0 = 3.5
            assert np.allclose(
                img[:, :, 2], 3.5, atol=1e-5
            ), "Channel 2 should be weighted average 3.5"

            # Process the image
            processed = _process_image(
                img,
                test_config,
            )

            # Check that processed image maintains the expected dimensions
            assert processed.shape == (
                test_config.size[0],
                test_config.size[1],
                test_config.n_output_channels,
            ), "Processed image should maintain 3 channels"

            # Verify the image was normalized correctly to uint8 range (0-255)
            # If normalisation is CONVERSION_ONLY, values are rescaled to 0-255 range
            # Channel 0 should be 0
            # Channel 1 should be 2.0/3.5*255 =~ 146
            # Channel 2 should be 3.5/3.5*255 =~ 255

            # Allow some tolerance due to normalization and rounding
            assert np.mean(processed[:, :, 0]) < 1, "Channel 0 should be close to 0"

            # Check channel 1 (should be around 146)
            mean_ch1 = np.mean(processed[:, :, 1])
            assert np.isclose(
                mean_ch1, 146, atol=2
            ), f"Channel 1 should be close to 146, got {mean_ch1}"

            # Check channel 2 (should be around 255)
            mean_ch2 = np.mean(processed[:, :, 2])
            assert np.isclose(
                mean_ch2, 255, atol=2
            ), f"Channel 2 should be close to 255, got {mean_ch2}"

        finally:
            # Clean up temporary file
            if os.path.exists(temp_fits_path):
                try:
                    os.remove(temp_fits_path)
                except (OSError, PermissionError):
                    pass
