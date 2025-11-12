# fitsbolt - A Python package for image loading and processing
# Copyright (C) <2025>  <Ruhberg>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the MIT License

"""
Tests for config validation and creation functions.
"""

import pytest
import numpy as np

from fitsbolt.cfg.create_config import create_config
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod


class TestConfigValidation:
    """Test class for config validation functions."""

    def test_validation_with_constraints(self):
        """Test parameter validation with various constraints through create_config."""
        # Test string validation with allowed values
        with pytest.raises(ValueError):
            create_config(log_level="INVALID_LEVEL")

        # Test numeric validation with min/max
        with pytest.raises(ValueError):
            create_config(interpolation_order=-1)

        with pytest.raises(ValueError):
            create_config(interpolation_order=6)

        # Test with invalid n_output_channels
        with pytest.raises(ValueError):
            create_config(n_output_channels=0)

        # Test with invalid normalisation_method
        with pytest.raises(ValueError):
            create_config(normalisation_method="NOT_A_METHOD")

        # Test with output_dtype that's not a numpy type
        with pytest.raises(ValueError):
            create_config(output_dtype=str)

    def test_create_config_edge_cases(self):
        """Test edge cases in create_config function."""
        # Test with None values
        cfg = create_config(
            size=None,  # Test with size=None (no resizing)
            fits_extension=None,
            normalisation_method=NormalisationMethod.CONVERSION_ONLY,
        )
        assert cfg.size is None

        # Test with invalid size format
        with pytest.raises(ValueError):
            create_config(size=123)  # Not a list or tuple

    def test_fits_extension_handling(self):
        """Test fits_extension handling in create_config."""
        # Test with integer fits_extension
        cfg = create_config(fits_extension=1)
        assert cfg.fits_extension == [1] or cfg.fits_extension == 1

        # Test with string fits_extension
        cfg = create_config(fits_extension="PRIMARY")
        assert cfg.fits_extension == ["PRIMARY"] or cfg.fits_extension == "PRIMARY"

        # Test with list fits_extension
        cfg = create_config(fits_extension=[0, 1, 2], n_output_channels=3)
        assert isinstance(cfg.fits_extension, list)
        assert len(cfg.fits_extension) == 3

        # Test with invalid combination (fits_extension list length != n_output_channels)
        with pytest.raises(ValueError):
            create_config(fits_extension=[0, 1, 2, 3], n_output_channels=3)

    def test_channel_combination_validation(self):
        """Test channel_combination validation in create_config."""
        # Test with custom channel_combination
        channel_comb = np.array([[1, 0], [0, 1]])
        cfg = create_config(
            fits_extension=[0, 1],
            n_output_channels=2,
            channel_combination=channel_comb,
        )
        assert np.array_equal(cfg.channel_combination, channel_comb)

        # Test with invalid shape
        with pytest.raises(ValueError):
            create_config(
                fits_extension=[0, 1],
                n_output_channels=2,
                channel_combination=np.array([[1, 0, 0], [0, 1, 0]]),  # Wrong shape (2x3)
            )

    def test_norm_parameters_validation(self):
        """Test validation of normalisation parameters."""
        # Test with invalid norm_asinh_scale
        with pytest.raises(ValueError):
            create_config(
                n_output_channels=3,
                norm_asinh_scale=[0.7, 0.7],  # Wrong length (should be 3)
            )

        # Test with invalid norm_asinh_clip
        with pytest.raises(ValueError):
            create_config(
                n_output_channels=3,
                norm_asinh_clip=[99.8, 99.8],  # Wrong length (should be 3)
            )

    def test_additional_config_combinations(self):
        """Test additional combinations of config parameters."""
        # Test with default parameters
        cfg = create_config()
        assert cfg.output_dtype == np.uint8
        assert cfg.size == [224, 224]
        assert cfg.fits_extension is None
        assert cfg.interpolation_order == 1
        assert cfg.n_output_channels == 3
        assert cfg.normalisation_method == NormalisationMethod.CONVERSION_ONLY
        assert cfg.num_workers == 4
        assert cfg.log_level in ["TRACE", "WARNING", "INFO", "DEBUG", "ERROR", "CRITICAL"]
        assert cfg.force_dtype is True

        # Test with custom output dtype
        cfg = create_config(output_dtype=np.float32)
        assert cfg.output_dtype == np.float32

        # Test with all custom parameters
        cfg = create_config(
            output_dtype=np.uint16,
            size=[128, 128],
            fits_extension=0,
            interpolation_order=3,
            n_output_channels=1,
            normalisation_method=NormalisationMethod.ASINH,  # Updated to use a valid enum value
            num_workers=2,
            norm_maximum_value=1000,
            norm_minimum_value=0,
            log_level="DEBUG",
            force_dtype=False,
        )

        assert cfg.output_dtype == np.uint16
        assert cfg.size == [128, 128]
        assert cfg.fits_extension == 0 or cfg.fits_extension == [0]
        assert cfg.interpolation_order == 3
        assert cfg.n_output_channels == 1
        assert cfg.normalisation_method == NormalisationMethod.ASINH
        assert cfg.num_workers == 2
        assert cfg.normalisation.maximum_value == 1000
        assert cfg.normalisation.minimum_value == 0
        assert cfg.log_level == "DEBUG"
        assert cfg.force_dtype is False
