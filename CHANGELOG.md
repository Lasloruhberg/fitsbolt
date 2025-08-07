# Changelog

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