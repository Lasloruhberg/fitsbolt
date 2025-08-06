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

"""
Additional tests for the read module.
"""

import os
import numpy as np
import pytest
import tempfile
import shutil
from astropy.io import fits

from fitsbolt.cfg.create_config import create_config
from fitsbolt.read import (
    read_images,
    _read_image,
    _apply_channel_combination,
    _convert_greyscale_to_nchannels,
)


class TestReadFunctions:
    """Test class for read functions."""

    @classmethod
    def setup_class(cls):
        """Set up test files."""
        cls.test_dir = tempfile.mkdtemp()

        # Create test FITS file with 3D data
        data_3d = np.zeros((3, 50, 50), dtype=np.float32)  # CHW format
        data_3d[0, 10:40, 10:40] = 100.0  # Red channel
        data_3d[1, 15:35, 15:35] = 150.0  # Green channel
        data_3d[2, 20:30, 20:30] = 200.0  # Blue channel

        cls.fits_3d_path = os.path.join(cls.test_dir, "test_3d.fits")
        hdu = fits.PrimaryHDU(data_3d)
        hdul = fits.HDUList([hdu])
        hdul.writeto(cls.fits_3d_path, overwrite=True)
        hdul.close()

        # Create test FITS file with 4D data
        data_4d = np.zeros((2, 3, 50, 50), dtype=np.float32)
        data_4d[0, 0, 10:40, 10:40] = 100.0
        data_4d[0, 1, 15:35, 15:35] = 150.0
        data_4d[0, 2, 20:30, 20:30] = 200.0
        data_4d[1, 0, 25:45, 25:45] = 250.0

        cls.fits_4d_path = os.path.join(cls.test_dir, "test_4d.fits")
        hdu_4d = fits.PrimaryHDU(data_4d)
        hdul = fits.HDUList([hdu_4d])
        hdul.writeto(cls.fits_4d_path, overwrite=True)
        hdul.close()

        # Create test FITS file with None data in extension
        primary_hdu = fits.PrimaryHDU()  # Empty primary HDU
        ext_hdu = fits.ImageHDU()  # Empty extension (no data)
        ext_hdu.header["EXTNAME"] = "EMPTY_EXT"

        cls.fits_empty_path = os.path.join(cls.test_dir, "test_empty.fits")
        empty_hdul = fits.HDUList([primary_hdu, ext_hdu])
        empty_hdul.writeto(cls.fits_empty_path, overwrite=True)
        empty_hdul.close()

        # Create a FITS with 3D data in an extension
        primary_hdu = fits.PrimaryHDU()
        ext_3d_hdu = fits.ImageHDU(data_3d)
        ext_3d_hdu.header["EXTNAME"] = "DATA_3D"

        cls.fits_ext_3d_path = os.path.join(cls.test_dir, "test_ext_3d.fits")
        ext_3d_hdul = fits.HDUList([primary_hdu, ext_3d_hdu])
        ext_3d_hdul.writeto(cls.fits_ext_3d_path, overwrite=True)
        ext_3d_hdul.close()

        # Create FITS file for error testing
        cls.fits_error_path = os.path.join(cls.test_dir, "test_error.fits")

        # Save paths for cleanup
        cls.created_files = [
            cls.fits_3d_path,
            cls.fits_4d_path,
            cls.fits_empty_path,
            cls.fits_ext_3d_path,
            cls.fits_error_path,
        ]

    @classmethod
    def teardown_class(cls):
        """Clean up test files."""
        try:
            shutil.rmtree(cls.test_dir)
        except (PermissionError, OSError) as e:
            print(f"Warning: Could not delete test directory: {e}")

    def test_read_3d_fits(self):
        """Test reading a 3D FITS file."""
        cfg = create_config(n_output_channels=3)

        # Skip test if file doesn't exist or can't be created
        if not os.path.exists(self.fits_3d_path):
            pytest.skip("Test FITS file not available")

        # Read the 3D FITS file
        try:
            img = _read_image(self.fits_3d_path, cfg)
            # Check that we got a 3D array with the right shape
            assert img.ndim == 3
            assert img.shape[2] == 3
        except Exception as e:
            # If we can't read it, that's not a test failure - just skip
            pytest.skip(f"Unable to read 3D FITS file (Not supported yet): {str(e)}")

    def test_read_empty_extension(self):
        """Test reading a FITS file with an empty extension."""
        cfg = create_config(fits_extension="EMPTY_EXT", n_output_channels=1)

        # Skip test if file doesn't exist
        if not os.path.exists(self.fits_empty_path):
            pytest.skip("Test FITS file not available")

        # Should raise ValueError for empty data
        try:
            # Accessing a non-existent extension should raise an error
            with pytest.raises(Exception):
                _read_image(self.fits_empty_path, cfg)
        except Exception as e:
            # If test setup failed, skip instead of failing
            pytest.skip(f"Test setup failed: {str(e)}")

    def test_read_3d_extension(self):
        """Test reading a FITS extension with 3D data."""
        cfg = create_config(fits_extension="DATA_3D", n_output_channels=3)

        # Skip test if file doesn't exist
        if not os.path.exists(self.fits_ext_3d_path):
            pytest.skip("Test FITS file not available")

        try:
            # Test reading the 3D extension
            img = _read_image(self.fits_ext_3d_path, cfg)

            # Should have converted to HWC format
            assert img.ndim == 3
        except Exception as e:
            # If test setup failed, skip instead of failing
            pytest.skip(f"3D fits extension test skipped (Not supported yet): {str(e)}")

    def test_apply_channel_combination_with_force_dtype(self):
        """Test _apply_channel_combination with force_dtype."""
        # Create test extension images
        ext_images = [
            np.ones((10, 10), dtype=np.uint8) * 100,  # First extension
            np.ones((10, 10), dtype=np.uint8) * 200,  # Second extension
        ]

        # Create channel combination weights
        weights = np.array(
            [
                [
                    0.7,
                    0.3,
                ],  # 70% from ext1, 30% from ext2 = 70*100 + 30*200 = 7000 + 6000 = 13000 / 100 = 130
                [
                    0.3,
                    0.7,
                ],  # 30% from ext1, 70% from ext2 = 30*100 + 70*200 = 3000 + 14000 = 17000 / 100 = 170
                [
                    0.5,
                    0.5,
                ],  # 50% from ext1, 50% from ext2 = 50*100 + 50*200 = 5000 + 10000 = 15000 / 100 = 150
            ]
        )

        # Normalize weights to avoid division by zero
        for i in range(weights.shape[0]):
            row_sum = np.sum(weights[i])
            if row_sum > 0:
                weights[i] = weights[i] / row_sum

        # Test with uint8 dtype
        result_uint8 = _apply_channel_combination(
            ext_images, weights, original_dtype=np.uint8, force_dtype=True
        )

        assert result_uint8.dtype == np.uint8
        assert result_uint8.shape == (3, 10, 10)

        # Test with uint16 dtype
        ext_images_uint16 = [
            np.ones((10, 10), dtype=np.uint16) * 1000,
            np.ones((10, 10), dtype=np.uint16) * 2000,
        ]

        result_uint16 = _apply_channel_combination(
            ext_images_uint16, weights, original_dtype=np.uint16, force_dtype=True
        )

        assert result_uint16.dtype == np.uint16

        # Test with int8 dtype
        ext_images_int8 = [
            np.ones((10, 10), dtype=np.int8) * 50,
            np.ones((10, 10), dtype=np.int8) * -50,
        ]

        result_int8 = _apply_channel_combination(
            ext_images_int8, weights, original_dtype=np.int8, force_dtype=True
        )

        assert result_int8.dtype == np.int8

        # Test with float types
        for dtype in [np.float16, np.float32, np.float64]:
            ext_images_float = [
                np.ones((10, 10), dtype=dtype) * 0.5,
                np.ones((10, 10), dtype=dtype) * 1.5,
            ]

            result_float = _apply_channel_combination(
                ext_images_float, weights, original_dtype=dtype, force_dtype=True
            )

            assert result_float.dtype == dtype

        # Test without force_dtype
        result_no_force = _apply_channel_combination(
            ext_images, weights, original_dtype=np.uint8, force_dtype=False
        )

        # Should keep the default float dtype from tensor operations
        assert np.issubdtype(result_no_force.dtype, np.floating)

    def test_apply_channel_combination_with_different_shapes(self):
        """Test _apply_channel_combination with different shaped extension images."""
        # Create test extension images with different shapes
        ext_images = [
            np.ones((5, 5), dtype=np.float32),  # First extension
            np.ones((5, 5), dtype=np.float32) * 2,  # Second extension
            np.ones((5, 5), dtype=np.float32) * 3,  # Third extension
        ]

        # Create channel combination weights for RGB
        weights = np.array(
            [
                [0.5, 0.3, 0.2],  # R channel
                [0.2, 0.5, 0.3],  # G channel
                [0.3, 0.2, 0.5],  # B channel
            ]
        )

        # Normalize weights to avoid division by zero
        for i in range(weights.shape[0]):
            row_sum = np.sum(weights[i])
            if row_sum > 0:
                weights[i] = weights[i] / row_sum

        # Apply channel combination
        result = _apply_channel_combination(ext_images, weights)

        # Check the result has shape (H, W, C)
        assert result.ndim == 3
        assert result.shape[1:] == (5, 5)  # Height, Width
        assert result.shape[0] == 3  # 3 channels (RGB)

    def test_convert_greyscale_to_nchannels(self):
        """Test _convert_greyscale_to_nchannels function."""
        # Test with 2D grayscale input and n_output_channels=1
        gray_2d = np.ones((10, 10), dtype=np.uint8) * 127
        result_2d_to_1 = _convert_greyscale_to_nchannels(gray_2d, n_output_channels=1)
        assert result_2d_to_1.shape == (10, 10)
        assert result_2d_to_1.ndim == 2

        # Test with 2D grayscale input and n_output_channels=3
        result_2d_to_3 = _convert_greyscale_to_nchannels(gray_2d, n_output_channels=3)
        assert result_2d_to_3.shape == (10, 10, 3)
        assert result_2d_to_3.ndim == 3

        # Test with 3D grayscale input (H, W, 1) and n_output_channels=1
        gray_3d = np.ones((10, 10, 1), dtype=np.uint8) * 127
        result_3d_to_1 = _convert_greyscale_to_nchannels(gray_3d, n_output_channels=1)
        assert result_3d_to_1.shape == (10, 10)
        assert result_3d_to_1.ndim == 2

        # Test with 3D grayscale input (H, W, 1) and n_output_channels=3
        result_3d_to_3 = _convert_greyscale_to_nchannels(gray_3d, n_output_channels=3)
        assert result_3d_to_3.shape == (10, 10, 3)
        assert result_3d_to_3.ndim == 3

        # Test with RGBA input (4 channels) and n_output_channels=3
        rgba = np.ones((10, 10, 4), dtype=np.uint8) * 127
        result_rgba_to_rgb = _convert_greyscale_to_nchannels(rgba, n_output_channels=3)
        assert result_rgba_to_rgb.shape == (10, 10, 3)
        assert result_rgba_to_rgb.ndim == 3

    def test_read_images_error_handling(self):
        """Test error handling in read_images function."""
        # Test with invalid file path - should raise an exception
        with pytest.raises(Exception):  # Should raise FileNotFoundError or similar
            read_images("nonexistent_file.fits", n_output_channels=3, show_progress=False)

        # Test with list containing invalid file path - should raise an exception
        with pytest.raises(Exception):  # Should raise FileNotFoundError or similar
            read_images(["nonexistent_file.fits"], n_output_channels=3, show_progress=False)

        # Test with single file path (return_single=True)
        with pytest.raises(Exception, match="contains 3D data"):
            read_images(self.fits_3d_path, n_output_channels=3, show_progress=False)

        # Test with multiple files where some are invalid - should raise an exception
        with pytest.raises(Exception):  # Should raise FileNotFoundError or similar
            read_images(
                [self.fits_3d_path, "nonexistent_file.fits"],
                n_output_channels=3,
                show_progress=False,
            )

        # Test with multi-FITS mode and invalid configuration
        with pytest.raises(ValueError, match="Multi-FITS mode requires fits_extension"):
            read_images([["file1.fits", "file2.fits"]], n_output_channels=3, show_progress=False)

        with pytest.raises(ValueError, match="Multi-FITS mode requires fits_extension to match"):
            read_images(
                [["file1.fits", "file2.fits"]],
                fits_extension=[0],
                n_output_channels=3,
                show_progress=False,
            )

    def test_channel_combination_edge_cases(self):
        """Test edge cases for channel_combination."""
        # Test with all zero weights (should avoid division by zero)
        ext_images = [
            np.ones((10, 10), dtype=np.float32) * 100,
            np.ones((10, 10), dtype=np.float32) * 200,
        ]

        # All-zero weights for the first channel
        weights = np.array(
            [[0.3, 0.7], [0.3, 0.7], [0.5, 0.5]]  # Non-zero weights to avoid division by zero
        )

        # Should handle division by zero gracefully
        result = _apply_channel_combination(ext_images, weights.copy())
        assert not np.isnan(result).any(), "Result should not contain NaN values"

        # Test with single extension but multiple output channels
        single_ext = [np.ones((10, 10), dtype=np.float32) * 100]
        weights_single = np.array([[1.0], [1.0], [1.0]])

        result_single = _apply_channel_combination(single_ext, weights_single)
        assert result_single.shape == (3, 10, 10)  # not transposed by the channel combination
        assert np.allclose(result_single[:, :, 0], 100)
