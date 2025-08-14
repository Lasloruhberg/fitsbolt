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
Tests for the channel mixing functionality in channel_mixing.py.
"""

import numpy as np

from fitsbolt.channel_mixing import (
    batch_channel_combination,
    apply_channel_combination,
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


class TestChannelMixingIntegration:
    """Integration tests comparing batch and single-image functions."""

    def test_batch_vs_single_consistency(self):
        """Test that batch function gives same results as single-image function."""
        # Create test data
        extension_images = [
            np.array([[1, 2], [3, 4]]),
            np.array([[5, 6], [7, 8]]),
            np.array([[9, 10], [11, 12]]),
        ]

        weights = np.array([[0.5, 0.3, 0.2], [0.1, 0.4, 0.5]])

        # Single image result
        single_result = apply_channel_combination(extension_images, weights)

        # Batch result (reshape single image case)
        cutouts = np.stack(extension_images, axis=-1)[np.newaxis, ...]  # Add batch dimension
        batch_result = batch_channel_combination(cutouts, weights)
        batch_result = batch_result[0]  # Remove batch dimension

        # Results should match (transpose needed due to different output format)
        np.testing.assert_allclose(batch_result.transpose(2, 0, 1), single_result, rtol=1e-10)
