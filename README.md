# astro_loader

A versatile Python package for loading and normalizing astronomical images across multiple formats (FITS, JPEG, PNG, TIFF). The package provides a uniform interface for image processing, particularly optimized for astronomical data with high dynamic range.

## Installation

```bash
# Install from PyPI
pip install astro_loader

# Install from source
pip install git+https://github.com/Lasloruhberg/astro_loader.git
```

## Quick Start

### Loading Multiple Images

```python
from astro_loader.image_loader import load_and_process_images
from astro_loader.normalisation.NormalisationMethod import NormalisationMethod
import numpy as np

# List of image paths
filepaths = ["image1.fits", "image2.fits", "image3.jpg"]

# Load and process images with default settings
results = load_and_process_images(
    filepaths=filepaths,
    size=[224, 224],                                    # Target size for resizing
    normalisation_method=NormalisationMethod.ASINH,     # Normalization method
    fits_extension=0,                                   # FITS extension to use
    num_workers=4,                                      # Parallel processing
    show_progress=True                                  # Show progress bar
)

# Results contains tuples of (filepath, processed_image)
for path, img in results:
    print(f"Loaded {path}, shape: {img.shape}, dtype: {img.dtype}")
```

### Custom Image Processing

For cases where you have your own image loading pipeline but want to use astro_loader's processing capabilities:

```python
import numpy as np
from astro_loader.image_loader import process_image
from astro_loader.cfg.create_config import create_config
from astro_loader.normalisation.NormalisationMethod import NormalisationMethod

# Load image data using your own method
raw_image = your_image_loading_function()

# Create configuration
cfg = create_config(
    size=[224, 224],
    normalisation_method=NormalisationMethod.ZSCALE,
    output_dtype=np.uint8,
    interpolation_order=1
)

# Process the image
processed_image = process_image(
    image=raw_image,
    cfg=cfg,
    convert_to_rgb=True,
    image_source="custom_source"
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

astro_loader provides advanced capabilities for working with multi-extension FITS files:

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

- **Explicit mapping**: A numpy array of shape `(n_output_channels, len(fits_extension))` 
  - Each row represents an output channel
  - Each column represents a weight for the corresponding FITS extension
  - Weights are normalized to sum to 1 for each output channel
  
**Example**: If you have 3 FITS extensions and want a custom RGB mapping:
```python
# Create a custom mapping: 
# R = 0.7*ext0 + 0.3*ext1
# G = 1.0*ext1
# B = 0.5*ext1 + 0.5*ext2
channel_map = np.array([
    [0.7, 0.3, 0.0],  # R channel
    [0.0, 1.0, 0.0],  # G channel
    [0.0, 0.5, 0.5]   # B channel
])

results = load_and_process_images(
    filepaths=filepaths,
    fits_extension=[0, 1, 2],
    n_output_channels=3,
    channel_combination=channel_map
)
```

### Normalization Methods

astro_loader provides several normalization methods for handling astronomical images with high dynamic range:

1. **CONVERSION_ONLY**:
   - If input dtype already matches the requested output dtype: No conversion applied
   - For specific direct conversions:
     - uint16 to uint8: Scaled by dividing by 65535.0
     - float32/float64 in [0,1] range to uint8: Direct conversion
     - uint8 to uint16: Scaled by dividing by 255.0
     - uint8/uint16 to float32: Scaled to [0,1] range
   - For all other cases: Linear stretch between min and max values

2. **LOG**:
   - Applies a logarithmic stretch
   - Minimum: Either 0 or the minimum value of the array (controlled by log_calculate_minimum_value)
   - Maximum: Determined dynamically or set by norm_maximum_value
   - Optional center cropping available for maximum value determination

3. **ZSCALE**:
   - Applies a linear stretch using the ZScale algorithm from astropy
   - Uses statistical sampling to determine optimal contrast limits
   - Falls back to CONVERSION_ONLY if min=max

4. **ASINH**:
   - Applies an inverse hyperbolic sine (arcsinh) stretch
   - Handles high dynamic range data especially well
   - Parameters:
     - First clips based on min/max values
     - Then applies channel-wise percentile clipping (controlled by asinh_clip)
     - Stretching intensity controlled by asinh_scale (lower values = more linear behavior, higher values = more logarithmic)

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

#### Normalization Parameters
- **normalisation_method**: Method to use for normalization (CONVERSION_ONLY, LOG, ZSCALE, ASINH)
- **norm_maximum_value**: Maximum value for normalization (overrides auto-detection)
- **norm_minimum_value**: Minimum value for normalization (overrides auto-detection)
- **norm_crop_for_maximum_value**: Tuple (height, width) to crop around center for max value calculation
- **norm_log_calculate_minimum_value**: Whether to calculate minimum for log scaling (default: False)
- **norm_asinh_scale**: Channel-wise stretch factors for asinh normalization (default: [0.7, 0.7, 0.7])
- **norm_asinh_clip**: Channel-wise percentile clipping for asinh normalization (default: [99.8, 99.8, 99.8])
