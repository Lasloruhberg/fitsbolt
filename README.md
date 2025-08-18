fitsbolt - A Python package for image loading and processing
Copyright (C) <2025>  <Ruhberg>

# fitsbolt

A versatile Python package for loading and normalising astronomical images across multiple formats (FITS, JPEG, PNG, TIFF). The package provides a uniform interface for image processing, particularly optimized for astronomical data with high dynamic range.

## Installation

```bash
# Install from PyPI
pip install fitsbolt

# Install from source
pip install git+https://github.com/Lasloruhberg/fitsbolt.git

# make conda env
conda env create -f environment.yml
conda activate fb
```

## Quick Start

Also see the interactive [example Notebook](examples/multi_extension_tutorial.ipynb)

### Using Individual Processing Functions

fitsbolt now provides individual functions for each processing step, allowing you to have more granular control over your image processing pipeline:

```python
import fitsbolt
import numpy as np

# List of image paths
filepaths = ["image1.fits", "image2.fits"] # could be jpg,tiff,png

# Step 1: Read images (read_only=True to skip channel combination)
raw_images = fitsbolt.read_images(
    filepaths=filepaths,
    fits_extension=0,
    n_output_channels=3, # RGB, H,W,C format
    num_workers=4,
    show_progress=True,
    read_only=True  # Skip channel combination for manual processing
)

# Step 2: Resize images (recommended to use float32 for processing)
resized_images = fitsbolt.resize_images(
    images=raw_images,
    size=[224, 224],
    interpolation_order=1,
    output_dtype=np.float32,  # Recommended for processing
    show_progress=True
)

# Step 3: Apply channel combination if needed
if raw_images[0].shape[-1] > 3:  # If more channels than needed
    combined_images = fitsbolt.batch_channel_combination(
        resized_images,
        channel_combination=np.eye(3),  # Identity matrix for direct mapping
        output_dtype=None  # Keep as float for normalization
    )
else:
    combined_images = resized_images

# Step 4: Normalise images  
final_images = fitsbolt.normalise_images(
    images=combined_images,
    normalisation_method=fitsbolt.NormalisationMethod.LOG,
    output_dtype=np.uint8,  # Final output as uint8
    show_progress=True
)

print(f"Processed {len(final_images)} images")
for i, img in enumerate(final_images):
    print(f"Image {i}: shape={img.shape}, dtype={img.dtype}")
```

### Using the Complete Pipeline

For convenience, images can be read and processed by just one function call with the recommended processing order (read, resize, combine channels, normalise):

```python
from fitsbolt import load_and_process_images, NormalisationMethod
import numpy as np

# List of image paths if multiple formats are provided they must have the same number of channels
filepaths = ["image1.fits", "image2.fits", "image3.jpg"]

# Load and process images with default settings
results = load_and_process_images(
    filepaths=filepaths,
    size=[224, 224],                                    # Target size for resizing
    normalisation_method=NormalisationMethod.ASINH,     # Normalisation method
    fits_extension=['SCI','ERR','WHT'],                 # FITS extension to use
    n_output_channels=3, # will map the primary extension into a grey RGB image
    num_workers=4,                                      # Parallel processing
    show_progress=True                                  # Show progress bar
)

# Results contains processed images
for i, img in enumerate(results):
    print(f"Loaded image {i}, shape: {img.shape}, dtype: {img.dtype}")
```

### Advanced Usage Examples
See the example [Notebook](examples/multi_extension_tutorial.ipynb) for examples on any fits import.
#### Processing Single Images

```python
import fitsbolt
import numpy as np

# Process a single image file
single_image = fitsbolt.read_images("galaxy.fits")[0]
normalised = fitsbolt.normalise_images([single_image], 
                                     normalisation_method=fitsbolt.NormalisationMethod.ZSCALE)[0]
resized = fitsbolt.resize_images([normalised], size=[512, 512])[0]
```

#### Custom Channel Combination for Multi-Extension FITS

```python
import fitsbolt
import numpy as np

# Custom channel mapping for RGB from 3 FITS extensions
channel_map = np.array([
    [0.7, 0.3, 0.0],  # R = 70% ext0 + 30% ext1
    [0.0, 1.0, 0.0],  # G = 100% ext1
    [0.0, 0.5, 1.5]   # B = 50% ext1 + 50% ext2
])

# Read multi-extension FITS files
multi_ext_images = fitsbolt.read_images(
    filepaths=["multi_ext.fits"],
    fits_extension=[0, 1, 2],
    channel_combination=channel_map
)
```

## API Overview

### Individual Processing Functions

fitsbolt provides three main processing functions that can be used independently:

#### `fitsbolt.read_images()`
Reads image files and returns raw image arrays.

```python
images = fitsbolt.read_images(
    filepaths=["img1.fits", "img2.jpg"],
    fits_extension=[0,1,2],           # FITS extension to use
    n_output_channels=3,        # 3 channels for both fits and jpg
    num_workers=4,              # Parallel processing
    show_progress=True,         # Progress bar
    read_only=False             # (Default), will combine channels base ond 
)
```

#### `fitsbolt.normalise_images()`
Normalises image arrays using various astronomical-optimized methods.

```python
normalised = fitsbolt.normalise_images(
    images=resized_images,
    normalisation_method=fitsbolt.NormalisationMethod.ASINH,
    norm_asinh_scale=[0.7, 0.7, 0.7],
    norm_asinh_clip=[99.8, 99.8, 99.8],
    output_dtype=np.uint8,     # Final output dtype
    num_workers=4,
    show_progress=True
)
```

#### `fitsbolt.resize_images()`
Resizes image arrays to specified dimensions.

```python
resized = fitsbolt.resize_images(
    images=raw_images,
    size=[224, 224],           # Target size [height, width]
    interpolation_order=1,     # 0-5, higher = smoother
    output_dtype=np.float32,   # Recommended: float32 for processing chain
    show_progress=True
)
```

### Complete Pipeline Function

#### `fitsbolt.load_and_process_images()`
Combines all steps in a single function call with the recommended processing order: read, resize, combine channels, normalise.

```python
processed = fitsbolt.load_and_process_images(
    filepaths=["img1.fits", "img2.jpg"],
    size=[224, 224],
    normalisation_method=fitsbolt.NormalisationMethod.ASINH,
    fits_extension=0,
    num_workers=4
)
```

#### `fitsbolt.batch_channel_combination()`
Applies channel combination to batch arrays when using manual processing.

```python
combined = fitsbolt.batch_channel_combination(
    images,                    # Input array (N, H, W, C)
    channel_combination,       # Combination matrix (n_out, n_in)  
    output_dtype=np.float32,   # Choose output dtype as computation converts to float32
)
```

## Image Processing Details

### Reading Process

The package supports multiple image formats:
- **FITS files**: Processed using astropy.io.fits
- **Standard formats** (JPG, PNG, TIFF): Processed using PIL

For FITS files, the package offers flexible extension handling:
- Single extension (by index or name)
- Multiple extensions (combined using channel_combination parameter)

### FITS Extension Handling

fitsbolt provides advanced capabilities for working with multi-extension FITS files:

#### `fits_extension` Parameter

This parameter controls which FITS extensions to load:

- **None**: Uses first extension (index 0)
- **Integer**: Specifies a single extension by index (e.g., `fits_extension=1`)
- **String**: Specifies a single extension by name (e.g., `fits_extension="SCI"`)
- **List**: Specifies multiple extensions to load (e.g., `fits_extension=[0, 1, 2]` or `fits_extension=["SCI", "VAR"]`)

#### `n_output_channels` Parameter

This parameter controls the number of channels in the output image (default is 3 for RGB images).

#### `channel_combination` Parameter

When loading multiple FITS extensions, this parameter controls how they are combined:

- **None**: Default mapping is applied:
  - If `len(fits_extension) == n_output_channels`: One-to-one mapping (identity matrix)
  - If `fits_extension` has only 1 element: Maps the single extension to all output channels
  - Otherwise: Raises an error if no explicit mapping is provided

- **Explicit mapping**: A numpy array of shape `(n_output_channels, n_channels_in_image)` 
  - Each row represents an output channel
  - Each column represents a weight for the corresponding FITS extension
  - Weights can be an array of any float, however it is recommended to only use positive weights

**Example**: If you have 3 FITS extensions and want a custom RGB mapping:
```python
# Create a custom mapping: 
# R = 0.7*ext0 + 0.3*ext1
# G = 1.0*ext1
# B = 0.5*ext1 + 0.5*ext2
channel_map = np.array([
    [0.7, 0.3, 0.0],  # R channel
    [0.0, 1.0, 0.0],  # G channel
    [0.0, 0.5, 1.5]   # B channel
])

results = load_and_process_images(
    filepaths=filepaths,
    fits_extension=[0, 1, 2],
    n_output_channels=3,
    channel_combination=channel_map
)
```

### Multi-FITS File Handling

fitsbolt supports combining multiple FITS files into a single image, where each file corresponds to a specific channel. This is useful when you have separate FITS files for different filters or observations that need to be combined.

#### Basic Multi-FITS Usage

```python
# Combine three separate FITS files into RGB channels
# Each file will be read with its corresponding extension
files_for_rgb = ["red_filter.fits", "green_filter.fits", "blue_filter.fits"]
extensions = [0, 0, 0]  # Use extension 0 from each file

# Multi-FITS mode: pass a list of lists
result = fitsbolt.load_and_process_images(
    filepaths=[files_for_rgb],  # List of lists!
    fits_extension=extensions,   # Must be a list matching file count
    n_output_channels=3,
    size=[224, 224]
)

# Or for multiple sets of multi-FITS files
multiple_sets = [
    ["set1_red.fits", "set1_green.fits", "set1_blue.fits"],
    ["set2_red.fits", "set2_green.fits", "set2_blue.fits"]
]
results = fitsbolt.load_and_process_images(
    filepaths=multiple_sets,
    fits_extension=[0, 0, 0],
    n_output_channels=3
)
```

#### Multi-FITS Requirements

- **File count must match extension count**: Each FITS file must have a corresponding extension specified
- **All files must be FITS format**: Mixed file types are not supported in multi-FITS mode
- **Consistent dimensions**: All FITS files must have the same spatial dimensions
- **Extension list required**: `fits_extension` must be a list when using multi-FITS mode

#### Multi-FITS vs Single-File Multi-Extension

```python
# Single FITS file with multiple extensions (existing functionality)
result = fitsbolt.load_and_process_images(
    filepaths=["multi_ext.fits"],     # Single file
    fits_extension=[0, 1, 2],         # Multiple extensions from same file
    n_output_channels=3
)

# Multiple FITS files with single extensions each (new functionality)
result = fitsbolt.load_and_process_images(
    filepaths=[["file1.fits", "file2.fits", "file3.fits"]],  # Multiple files
    fits_extension=[0, 1, 2],         # One extension per file
    n_output_channels=3
)
```

### Normalisation Methods

fitsbolt provides several normalisation methods for handling astronomical images with high dynamic range:

1. **CONVERSION_ONLY**:
   - If input dtype already matches the requested output dtype: No conversion applied
   - For specific direct conversions:
     - uint16 to uint8: Scaled by dividing by 65535.0
     - float32/float64 in [0,1] range to uint8: Direct conversion
     - uint8 to uint16: Scaled by dividing by 255.0
     - uint8/uint16 to float32: Scaled to [0,1] range
   - For all other cases: Linear stretch between min and max values

2. **LINEAR**:
   - Applies a linear stretch using astropy's LinearStretch
   - Uses the configured minimum and maximum values for normalisation
   - If not specified, uses the data's min/max values
   - Clips values to the target range

3. **LOG**:
   - Applies a logarithmic stretch
   - Minimum: Either 0 or the minimum value of the array (controlled by log_calculate_minimum_value)
   - Maximum: Determined dynamically or set by norm_maximum_value
   - Optional center cropping available for maximum value determination

4. **ZSCALE**:
   - Applies a linear stretch using the ZScale algorithm from astropy
   - Uses statistical sampling to determine optimal contrast limits
   - Falls back to CONVERSION_ONLY if min=max

5. **ASINH**:
   - Applies an inverse hyperbolic sine (arcsinh) stretch
   - Parameters:
      - First: clipping based on the set min/max, then clipping symmetrically by percentile each channel and then dividing with the asinh_scale cfg parameter, and normalising
      - Minimum and Maximum are computed or based on the norm_minimum/maximum_value parameters if set

6. **MIDTONES**:
   - Applies a Midtones Transfer Function (MTF) for fine-tuned contrast control
   - Similar to the "curves" tool in image editing software
   - Automatically computes the curve parameter to achieve a desired mean brightness
   - Parameters:
      - norm_midtones_percentile: Percentile for clipping (default: 99.8)
      - norm_midtones_desired_mean: Target mean brightness value (default: 0.2)
      - norm_midtones_crop: Optional crop area for mean calculation (h,w)

### Configuration Parameters

#### General Parameters
- **output_dtype**: Data type for output images (default: np.uint8)
- **size**: Target size for resizing [height, width]
- **interpolation_order**: Order of interpolation for resizing (0-5, default: 1)
- **num_workers**: Number of worker threads for parallel loading

#### FITS File Parameters
- **fits_extension**: FITS extension(s) to use (None, index, name, or list)
- **n_output_channels**: Number of channels in the output image (default: 3 for RGB)
- **channel_combination**: Array defining how to combine FITS extensions into output channels
- **read_only**: If True, skips automatic channel combination for manual processing (default: False)

#### Normalisation Parameters
- **normalisation_method**: Method to use for normalisation (CONVERSION_ONLY, LINEAR, LOG, ZSCALE, ASINH, MIDTONES)
- **norm_maximum_value**: Maximum value for normalisation (overrides auto-detection)
- **norm_minimum_value**: Minimum value for normalisation (overrides auto-detection)
- **norm_crop_for_maximum_value**: Tuple (height, width) to crop around center for max value calculation

##### Log Normalisation Parameters
- **norm_log_calculate_minimum_value**: Whether to calculate minimum for log scaling (default: False)
- **norm_log_scale_a**: Scale factor 'a' for astropy LogStretch (default: 1000.0)

##### Asinh Normalisation Parameters
- **norm_asinh_scale**: Channel-wise stretch factors for asinh normalisation (default: \[0.7\])
- **norm_asinh_clip**: Channel-wise percentile clipping for asinh normalisation (default: \[99.8\])

##### ZScale Normalisation Parameters
- **norm_zscale_n_samples**: Number of samples for zscale normalisation (default: 1000)
- **norm_zscale_contrast**: Contrast for zscale normalisation (default: 0.25)
- **norm_zscale_max_reject**: Maximum rejection fraction for zscale normalisation (default: 0.5)
- **norm_zscale_min_pixels**: Minimum number of pixels that must remain after rejection (default: 5)
- **norm_zscale_krej**: Number of sigma used for rejection (default: 2.5)
- **norm_zscale_max_iter**: Maximum number of iterations for zscale normalisation (default: 5)

##### Midtones Normalisation Parameters
- **norm_midtones_percentile**: Percentile for clipping in each channel (default: 99.8)
- **norm_midtones_desired_mean**: Target mean brightness value between 0 and 1 (default: 0.2)
- **norm_midtones_crop**: Optional crop dimensions (height, width) for calculating the mean
