# Changelog
## [0.1.2] - 2025-08-14

### Added

- Option to skip channel combiation in read_images (read_only = True)
- A function `batch_channel_combination`, to handle channel combinations

### Fixed

- Included full parameter control for zscale and log normalisation

### Changed

- Modularised the channel combination funciton into a standalone function `batch_channel_combination`
    - This function takes an np array (n_images, H, W, C) and combines them based on the 
- Changed default processing order to read, resize, normalise, combine

### Removed

- `_apply_channel_combination` function, replaced with `batch_channel_combination`
- `_convert_greyscale_to_nchannels` function, which is now incorporated into `batch_channel_combination`

## [0.1.1] - 2025-08-07

### Added

- multiple tests for read edge cases

### Fixed

- logger name set to 'fitsbolt'
- log level can be set at function call

### Changed

- logger is now module specific and does not remove other loggers.
- any failure in read, normalise or resize will lead to an Exception instead of just logging a warning

### Removed

## [0.1.0] - 2025-08-01

- intial publication

## Version overview
[0.1.1]: https://github.com/Lasloruhberg/fitsbolt/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Lasloruhberg/fitsbolt/releases/tag/v0.1.0