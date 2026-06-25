# Changelog
## [0.2.1] - 2026-06-25

### Added
- Possibility to determine vmin/vmax via sampling for Asinh instead.
- Channel combination can now be applied to non-fits images

### Fixed
- Improved Readme to include clipping and truncation of channel combination


## [0.2.0] - 2026-01-12

### Added
- Ability to choose between ThreadPools and ProcessPools
- Performance Improvements for normal png, jpg images and simple channel combinations

### Fixed
- Midtones now supports constant image channels
- Float32 Tiffles being cast to unit8 during read is now fixed.

## [0.1.6] - 2026-01-12

### Added
- Lazy loading to reduced import times

## [0.1.5] - 2025-11-19

### Changed
- Switch license to dual MIT/GLP-3.0

## [0.1.4] - 2025-11-12

### Changed
- Switch license to MIT

## [0.1.3.1] - 2025-09-26

### Changed
- Hotfix to remove print statements in the code.

## [0.1.3] - 2025-08-18

### Changed

- Updated processing order in `_process_image` to resize → combine channels → normalise for improved quality and consistency with manual workflow recommendations
- Added support for Python 3.10

## [0.1.2] - 2025-08-14

### Added

- Option to skip channel combiation in read_images (read_only = True)
- A function `batch_channel_combination`, to handle channel combinations
- Added Midtones Transfer Function (MIDTONES) normalisation method for better control over image contrast

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
[0.2.1]: https://github.com/Lasloruhberg/fitsbolt/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Lasloruhberg/fitsbolt/compare/v0.1.6...v0.2.0
[0.1.6]: https://github.com/Lasloruhberg/fitsbolt/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/Lasloruhberg/fitsbolt/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/Lasloruhberg/fitsbolt/compare/v0.1.3.1...v0.1.4
[0.1.3]: https://github.com/Lasloruhberg/fitsbolt/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/Lasloruhberg/fitsbolt/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Lasloruhberg/fitsbolt/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Lasloruhberg/fitsbolt/releases/tag/v0.1.0
