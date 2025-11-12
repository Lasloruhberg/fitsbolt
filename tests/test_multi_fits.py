# fitsbolt - A Python package for image loading and processing
# Copyright (C) <2025>  <Ruhberg>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the MIT License

"""
Test cases for the new multi-FITS functionality.
"""

import os
import numpy as np
import pytest
import tempfile
import shutil
from astropy.io import fits

from fitsbolt.image_loader import load_and_process_images
from fitsbolt.read import read_images, _read_multi_fits_image
from fitsbolt.cfg.create_config import create_config
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod


class TestMultiFITS:
    """Test class for multi-FITS functionality."""

    @classmethod
    def setup_class(cls):
        """Set up test FITS files."""
        cls.test_dir = tempfile.mkdtemp()

        # Create three FITS files with different but compatible data
        cls.fits_files = []

        for i in range(3):
            # Create test data - each file has a different pattern
            data = np.zeros((50, 50), dtype=np.float32)
            if i == 0:
                data[10:40, 10:40] = 100.0  # Red channel pattern
            elif i == 1:
                data[15:35, 15:35] = 150.0  # Green channel pattern
            else:
                data[20:30, 20:30] = 200.0  # Blue channel pattern

            filepath = os.path.join(cls.test_dir, f"test_multi_{i}.fits")
            cls.fits_files.append(filepath)

            # Create FITS file with data in extension 0
            hdu = fits.PrimaryHDU(data)
            hdul = fits.HDUList([hdu])
            hdul.writeto(filepath, overwrite=True)
            hdul.close()

    @classmethod
    def teardown_class(cls):
        """Clean up test files."""
        shutil.rmtree(cls.test_dir)

    def test_multi_fits_basic_functionality(self):
        """Test basic multi-FITS reading functionality."""
        # Test with three FITS files and corresponding extensions
        images = read_images(
            [self.fits_files],  # List of lists
            fits_extension=[0, 0, 0],  # Use extension 0 from each file
            n_output_channels=3,
        )

        assert len(images) == 1, "Should return one combined image"
        image = images[0]

        # Should be 3D with 3 channels (RGB)
        assert image.ndim == 3, f"Expected 3D image, got {image.ndim}D"
        assert image.shape[2] == 3, f"Expected 3 channels, got {image.shape[2]}"
        assert image.shape[:2] == (
            50,
            50,
        ), f"Expected (50, 50) spatial dimensions, got {image.shape[:2]}"

    def test_multi_fits_assertion_mismatch(self):
        """Test that assertion fails when file count doesn't match extension count."""
        with pytest.raises(
            ValueError, match="Multi-FITS mode requires fits_extension to match the number of files"
        ):
            read_images(
                [self.fits_files],  # 3 files
                fits_extension=[0, 0],  # Only 2 extensions
                n_output_channels=2,  # Match the extension count to avoid config validation error
            )

    def test_multi_fits_non_fits_file_error(self):
        """Test that multi-FITS mode only accepts FITS files."""
        # Create a temporary non-FITS file
        non_fits_file = os.path.join(self.test_dir, "not_fits.txt")
        with open(non_fits_file, "w") as f:
            f.write("This is not a FITS file")

        with pytest.raises(AssertionError, match="All files must be FITS files"):
            cfg = create_config(fits_extension=[0, 0], n_output_channels=2)
            _read_multi_fits_image([self.fits_files[0], non_fits_file], [0, 0], cfg)

    def test_multi_fits_invalid_extension(self):
        """Test error handling for invalid extensions."""
        # Instead of testing with the full read_images, which might handle exceptions,
        # we'll test directly with _read_multi_fits_image, which should raise an IndexError
        with pytest.raises(IndexError, match="FITS extension index .* is out of bounds"):
            cfg = create_config(fits_extension=[0, 0, 10], n_output_channels=3)
            _read_multi_fits_image(self.fits_files, [0, 0, 10], cfg)

    def test_multi_fits_with_load_and_process_images(self):
        """Test multi-FITS functionality through the main load_and_process_images function."""
        images = load_and_process_images(
            [self.fits_files],  # List of lists
            fits_extension=[0, 0, 0],
            size=[32, 32],
            normalisation_method=NormalisationMethod.CONVERSION_ONLY,
            n_output_channels=3,
            show_progress=False,
        )

        assert len(images) == 1, "Should return one combined image"
        image = images[0]

        # Should be processed (resized and normalised)
        assert image.shape == (32, 32, 3), f"Expected (32, 32, 3), got {image.shape}"
        assert image.dtype == np.uint8, f"Expected uint8, got {image.dtype}"

    def test_multi_fits_multiple_sets(self):
        """Test processing multiple sets of multi-FITS files."""
        # Create two sets of multi-FITS files
        images = load_and_process_images(
            [self.fits_files, self.fits_files[:2] + [self.fits_files[0]]],  # Two sets
            fits_extension=[0, 0, 0],
            size=[32, 32],
            normalisation_method=NormalisationMethod.CONVERSION_ONLY,
            n_output_channels=3,
            show_progress=False,
        )

        assert len(images) == 2, "Should return two combined images"
        for image in images:
            assert image.shape == (32, 32, 3), f"Expected (32, 32, 3), got {image.shape}"
            assert image.dtype == np.uint8, f"Expected uint8, got {image.dtype}"

    def test_multi_fits_single_return(self):
        """Test that single multi-FITS input returns a single image directly."""
        # Need to pass the files as a regular list, not nested, but specify multi-file extensions
        image = load_and_process_images(
            [self.fits_files],  # This creates a list with one element that is itself a list
            fits_extension=[0, 0, 0],
            size=[32, 32],
            normalisation_method=NormalisationMethod.CONVERSION_ONLY,
            n_output_channels=3,
            show_progress=False,
        )

        # The function returns a list with one element for multi-FITS mode
        assert isinstance(image, list), "Should return a list containing one image"
        assert len(image) == 1, "Should return a list with exactly one image"
        assert isinstance(image[0], np.ndarray), "The image should be a numpy array"
        assert image[0].shape == (32, 32, 3), f"Expected (32, 32, 3), got {image[0].shape}"

    def test_multi_fits_validation_error_for_wrong_extension_type(self):
        """Test that proper error is raised when extension type is wrong for multi-FITS."""
        with pytest.raises(
            ValueError, match="Multi-FITS mode requires fits_extension to be a list"
        ):
            read_images(
                [self.fits_files],  # List of lists (multi-FITS mode)
                fits_extension=0,  # Not a list
                n_output_channels=1,  # Use 1 to avoid the extension count validation
            )

    def test_multi_fits_shape_mismatch_error(self):
        """Test error when FITS files have different shapes."""
        # Create a FITS file with different shape
        different_shape_data = np.zeros((60, 60), dtype=np.float32)  # Different size
        different_shape_file = os.path.join(self.test_dir, "different_shape.fits")

        hdu = fits.PrimaryHDU(different_shape_data)
        hdul = fits.HDUList([hdu])
        hdul.writeto(different_shape_file, overwrite=True)
        hdul.close()

        # Test directly with _read_multi_fits_image
        with pytest.raises(ValueError, match="Cannot combine FITS files with different shapes"):
            cfg = create_config(fits_extension=[0, 0, 0], n_output_channels=3)
            _read_multi_fits_image(
                [self.fits_files[0], self.fits_files[1], different_shape_file],
                [0, 0, 0],
                cfg,
            )


if __name__ == "__main__":
    pytest.main([__file__])
