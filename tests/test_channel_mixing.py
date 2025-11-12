# fitsbolt - A Python package for image loading and processing
# Copyright (C) <2025>  <Ruhberg>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the MIT License

"""
Tests for the channel mixing functionality in channel_mixing.py.
"""

import numpy as np

from fitsbolt.channel_mixing import (
    batch_channel_combination,
)


class TestBatchChannelCombination:
    """Test cases for batch_channel_combination function."""

    def test_simple_single_image_single_channel(self):
        """Test with single image, single output channel."""
        # Create test data: 1 image, 2x2 pixels, 3 input channels
        cutouts = np.array(
            [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]
        )  # Shape: (1, 2, 2, 3)
        assert cutouts.shape == (1, 2, 2, 3)
        # Weights: 1 output channel, 3 input channels
        weights = np.array([[0.5, 0.3, 0.2]])  # Shape: (1, 3)

        result = batch_channel_combination(cutouts, weights)

        # Expected result for each pixel: 0.5*input[0] + 0.3*input[1] + 0.2*input[2]
        expected = np.array(
            [
                [
                    [[0.5 * 1 + 0.3 * 2 + 0.2 * 3], [0.5 * 4 + 0.3 * 5 + 0.2 * 6]],
                    [[0.5 * 7 + 0.3 * 8 + 0.2 * 9], [0.5 * 10 + 0.3 * 11 + 0.2 * 12]],
                ]
            ]
        )  # Shape: (1, 2, 2, 1)

        assert result.shape == (1, 2, 2, 1)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_multiple_images_multiple_channels(self):
        """Test with multiple images and multiple output channels."""
        # Create test data: 2 images, 2x2 pixels, 3 input channels
        cutouts = np.array(
            [
                [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],  # Image 1
                [[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]],  # Image 2
            ]
        )  # Shape: (2, 2, 2, 3)
        assert cutouts.shape == (2, 2, 2, 3)

        # Weights: 2 output channels, 3 input channels
        weights = np.array(
            [
                [1.0, 0.0, 0.0],  # Output channel 0: only first input channel
                [0.0, 0.0, 1.0],  # Output channel 1: only third input channel
            ]
        )  # Shape: (2, 3)

        result = batch_channel_combination(cutouts, weights)

        # Expected: first output channel = first input channel, second output = third input
        expected = np.array(
            [
                [[[1, 3], [4, 6]], [[7, 9], [10, 12]]],  # Image 1
                [[[13, 15], [16, 18]], [[19, 21], [22, 24]]],  # Image 2
            ]
        )  # Shape: (2, 2, 2, 2)

        assert result.shape == (2, 2, 2, 2)
        np.testing.assert_allclose(result, expected)

    def test_weighted_combination(self):
        """Test weighted combination without normalisation."""
        # Create simple test case
        cutouts = np.array([[[[10, 20]]]])  # Shape: (1, 1, 1, 2)
        weights = np.array([[2.0, 3.0]])  # Shape: (1, 2)

        result = batch_channel_combination(cutouts, weights)

        # Expected: 2.0 * 10 + 3.0 * 20 = 20 + 60 = 80
        expected = np.array([[[[80]]]])

        assert result.shape == (1, 1, 1, 1)
        np.testing.assert_allclose(result, expected)

    def test_dtype_preservation(self):
        """Test that original dtype is preserved when force_dtype=True."""
        cutouts = np.array([[[[1, 2]]]], dtype=np.uint8)
        weights = np.array([[0.5, 0.5]], dtype=np.float32)

        result = batch_channel_combination(cutouts, weights, output_dtype=np.uint8)

        assert result.dtype == np.uint8
        expected = np.array(
            [[[[1]]]], dtype=np.uint8
        )  # (0.5*1 + 0.5*2) = 1.5 -> 1 when cast to uint8
        np.testing.assert_array_equal(result, expected)

    def test_no_dtype_forcing(self):
        """Test that dtype is not forced when force_dtype=False."""
        cutouts = np.array([[[[1, 2]]]], dtype=np.uint8)
        weights = np.array([[0.5, 0.5]], dtype=np.float32)

        result = batch_channel_combination(cutouts, weights)

        # Result should be in float (from the computation)
        assert result.dtype != np.uint8
        expected = np.array([[[[1.5]]]])  # 0.5*1 + 0.5*2 = 1.5
        np.testing.assert_allclose(result, expected)

    def test_zero_weights(self):
        """Test behavior with zero weights (should not normalise)."""
        cutouts = np.array([[[[10, 20]]]])  # Shape: (1, 1, 1, 2)
        weights = np.array([[0.0, 0.0]])  # Shape: (1, 2)

        result = batch_channel_combination(cutouts, weights)

        # Expected: 0.0 * 10 + 0.0 * 20 = 0
        expected = np.array([[[[0]]]])

        np.testing.assert_allclose(result, expected)

    def test_single_channel_to_rgb(self):
        """Test converting single channel to RGB-like output."""
        # Single input channel to 3 output channels
        cutouts = np.array([[[[5]]]])  # Shape: (1, 1, 1, 1)
        weights = np.array(
            [[1.0], [0.8], [0.6]]  # Red channel  # Green channel  # Blue channel
        )  # Shape: (3, 1)

        result = batch_channel_combination(cutouts, weights)

        expected = np.array([[[[5, 4, 3]]]])  # [1.0*5, 0.8*5, 0.6*5]

        assert result.shape == (1, 1, 1, 3)
        np.testing.assert_allclose(result, expected)

    def test_edge_case_empty_weights(self):
        """Test edge case with empty output channels."""
        cutouts = np.array([[[[1, 2]]]])  # Shape: (1, 1, 1, 2)
        weights = np.array([]).reshape(0, 2)  # Shape: (0, 2)

        result = batch_channel_combination(cutouts, weights)

        assert result.shape == (1, 1, 1, 0)

    def test_tensor_axes_correctness(self):
        """Test that tensordot axes parameter is correct."""
        # This test verifies the tensor operation is working as expected
        cutouts = np.array(
            [
                [[[1, 10]], [[100, 1000]]],  # Image 1: clear pattern
                [[[2, 20]], [[200, 2000]]],  # Image 2: doubled pattern
            ]
        )  # Shape: (2, 2, 1, 2)

        weights = np.array(
            [[1, 0], [0, 1], [0, 1]]  # Take only first channel  # Take only second channel
        )  # Shape: (2, 2)

        result = batch_channel_combination(cutouts, weights)

        # Should extract the two input channels separately
        expected = np.array(
            [
                [[[1, 10, 10]], [[100, 1000, 1000]]],  # Image 1: [first_ch, second_ch, third_ch]
                [[[2, 20, 20]], [[200, 2000, 2000]]],  # Image 2: [first_ch, second_ch, third_ch]
            ]
        )  # Shape: (2, 2, 1, 3)

        assert result.shape == (2, 2, 1, 3)
        np.testing.assert_allclose(result, expected)

    def test_4_to_1_channel_combination(self):
        """Test converting 4 input channels to 1 output channel."""
        # RGBA to Grayscale conversion
        cutouts = np.array([[[[255, 128, 64, 255]]]])  # Shape: (1, 1, 1, 4)
        # Standard grayscale conversion weights (ignoring alpha)
        weights = np.array([[0.299, 0.587, 0.114, 0.0]])  # Shape: (1, 4)

        result = batch_channel_combination(cutouts, weights)

        expected_gray = 0.299 * 255 + 0.587 * 128 + 0.114 * 64 + 0.0 * 255
        expected = np.array([[[[expected_gray]]]])

        assert result.shape == (1, 1, 1, 1)
        np.testing.assert_allclose(result, expected)

    def test_4_to_3_channel_combination(self):
        """Test converting 4 input channels to 3 output channels (RGBA to RGB)."""
        cutouts = np.array([[[[100, 150, 200, 128]]]])  # Shape: (1, 1, 1, 4) - RGBA
        # Remove alpha channel, keep RGB
        weights = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],  # Red channel
                [0.0, 1.0, 0.0, 0.0],  # Green channel
                [0.0, 0.0, 1.0, 0.0],  # Blue channel
            ]
        )  # Shape: (3, 4)

        result = batch_channel_combination(cutouts, weights)

        expected = np.array([[[[100, 150, 200]]]])  # Just RGB, alpha removed

        assert result.shape == (1, 1, 1, 3)
        np.testing.assert_array_equal(result, expected)

    def test_dtype_uint8_clipping(self):
        """Test uint8 dtype conversion with clipping."""
        cutouts = np.array([[[[200, 100]]]], dtype=np.float32)
        weights = np.array([[1.5, 1.0]])  # 1.5*200 + 1.0*100 = 400, clips to 255

        result = batch_channel_combination(cutouts, weights, output_dtype=np.uint8)

        assert result.dtype == np.uint8
        expected = np.array([[[[255]]]], dtype=np.uint8)  # Clipped from 400 to 255
        np.testing.assert_array_equal(result, expected)

    def test_dtype_uint16_clipping(self):
        """Test uint16 dtype conversion with clipping."""
        cutouts = np.array([[[[60000, 10000]]]], dtype=np.float32)
        weights = np.array([[1.2, 0.0]])  # 1.2*60000 = 72000, clips to 65535

        result = batch_channel_combination(cutouts, weights, output_dtype=np.uint16)

        assert result.dtype == np.uint16
        expected = np.array([[[[65535]]]], dtype=np.uint16)  # Clipped from 72000 to 65535
        np.testing.assert_array_equal(result, expected)

    def test_dtype_int8_clipping(self):
        """Test int8 dtype conversion with clipping."""
        cutouts = np.array([[[[100]]]], dtype=np.float32)
        weights = np.array([[2.0]])  # 2.0*100 = 200, clips to 127 (max int8)

        result = batch_channel_combination(cutouts, weights, output_dtype=np.int8)

        assert result.dtype == np.int8
        expected = np.array([[[[127]]]], dtype=np.int8)
        np.testing.assert_array_equal(result, expected)

    def test_dtype_int16_clipping(self):
        """Test int16 dtype conversion with clipping."""
        cutouts = np.array([[[[-20000]]]], dtype=np.float32)
        weights = np.array([[2.0]])  # 2.0*(-20000) = -40000, clips to -32768 (min int16)

        result = batch_channel_combination(cutouts, weights, output_dtype=np.int16)

        assert result.dtype == np.int16
        expected = np.array([[[[-32768]]]], dtype=np.int16)
        np.testing.assert_array_equal(result, expected)

    def test_dtype_int32_clipping(self):
        """Test int32 dtype conversion with clipping."""
        cutouts = np.array([[[[2147483647]]]], dtype=np.float64)  # Near max int32
        weights = np.array([[1.1]])  # Will overflow int32

        result = batch_channel_combination(cutouts, weights, output_dtype=np.int32)

        assert result.dtype == np.int32
        expected = np.array([[[[2147483647]]]], dtype=np.int32)  # Clipped to max int32
        np.testing.assert_array_equal(result, expected)

    def test_dtype_int64_clipping(self):
        """Test int64 dtype conversion with clipping."""
        cutouts = np.array([[[[1000]]]], dtype=np.float64)  # Smaller safe value
        weights = np.array([[2.0]])

        result = batch_channel_combination(cutouts, weights, output_dtype=np.int64)

        assert result.dtype == np.int64
        expected = np.array([[[[2000]]]], dtype=np.int64)
        np.testing.assert_array_equal(result, expected)

    def test_dtype_float16_conversion(self):
        """Test float16 dtype conversion."""
        cutouts = np.array([[[[1.5]]]], dtype=np.float64)
        weights = np.array([[2.0]])

        result = batch_channel_combination(cutouts, weights, output_dtype=np.float16)

        assert result.dtype == np.float16
        expected = np.array([[[[3.0]]]], dtype=np.float16)
        np.testing.assert_array_equal(result, expected)

    def test_dtype_float32_conversion(self):
        """Test float32 dtype conversion."""
        cutouts = np.array([[[[1.5]]]], dtype=np.float64)
        weights = np.array([[2.0]])

        result = batch_channel_combination(cutouts, weights, output_dtype=np.float32)

        assert result.dtype == np.float32
        expected = np.array([[[[3.0]]]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_dtype_float64_conversion(self):
        """Test float64 dtype conversion."""
        cutouts = np.array([[[[1.5]]]], dtype=np.float32)
        weights = np.array([[2.0]])

        result = batch_channel_combination(cutouts, weights, output_dtype=np.float64)

        assert result.dtype == np.float64
        expected = np.array([[[[3.0]]]], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_dtype_custom_conversion(self):
        """Test custom dtype conversion (fallback case)."""
        cutouts = np.array([[[[1 + 2j]]]], dtype=np.complex128)
        weights = np.array([[2.0]])

        result = batch_channel_combination(cutouts, weights, output_dtype=np.complex64)

        assert result.dtype == np.complex64
        expected = np.array([[[[2 + 4j]]]], dtype=np.complex64)
        np.testing.assert_array_equal(result, expected)

    def test_no_dtype_conversion_when_same(self):
        """Test that no conversion happens when dtypes already match."""
        cutouts = np.array([[[[1.0]]]], dtype=np.float32)
        weights = np.array([[2.0]], dtype=np.float32)

        result = batch_channel_combination(cutouts, weights, output_dtype=np.float32)

        # Should not trigger conversion since combined already has the target dtype
        assert result.dtype == np.float32
        expected = np.array([[[[2.0]]]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_large_batch_multiple_shapes(self):
        """Test with larger batch and different image dimensions."""
        # 5 images, 3x4 pixels, 2 input channels -> 3 output channels
        cutouts = np.random.rand(5, 3, 4, 2).astype(np.float32)
        weights = np.array(
            [
                [1.0, 0.0],  # First output: copy first input
                [0.0, 1.0],  # Second output: copy second input
                [0.5, 0.5],  # Third output: average of inputs
            ]
        )

        result = batch_channel_combination(cutouts, weights)

        assert result.shape == (5, 3, 4, 3)

        # Check that first output channel equals first input channel
        np.testing.assert_array_equal(result[:, :, :, 0], cutouts[:, :, :, 0])
        # Check that second output channel equals second input channel
        np.testing.assert_array_equal(result[:, :, :, 1], cutouts[:, :, :, 1])
        # Check that third output channel is average of inputs
        expected_avg = (cutouts[:, :, :, 0] + cutouts[:, :, :, 1]) / 2
        np.testing.assert_allclose(result[:, :, :, 2], expected_avg)

    def test_zero_weight_combinations(self):
        """Test various zero weight combinations."""
        cutouts = np.array([[[[10, 20, 30]]]])  # Shape: (1, 1, 1, 3)

        # Different zero patterns
        weights = np.array(
            [
                [0.0, 0.0, 1.0],  # Only third channel
                [1.0, 0.0, 0.0],  # Only first channel
                [0.0, 0.0, 0.0],  # All zeros
            ]
        )

        result = batch_channel_combination(cutouts, weights)

        expected = np.array([[[[30, 10, 0]]]])  # [30, 10, 0]

        assert result.shape == (1, 1, 1, 3)
        np.testing.assert_allclose(result, expected)
