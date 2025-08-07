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

import sys
import logging
from typing import Optional


class FitsboltLogger:
    """Module-specific logger for fitsbolt with consistent formatting."""

    def __init__(self):
        self._logger_name = "fitsbolt"
        self._logger = logging.getLogger(self._logger_name)
        self._handler: Optional[logging.Handler] = None
        self._current_level = "INFO"

        # Prevent adding multiple handlers
        if not self._logger.handlers:
            self.set_log_level("INFO")

    def set_log_level(self, level: str):
        """Set the log level for the fitsbolt logger.

        Args:
            level (str): Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        # Remove existing handler if present
        if self._handler is not None:
            self._logger.removeHandler(self._handler)

        # Create custom formatter with fitsbolt-specific format
        formatter = FitsboltFormatter()

        # Create handler
        self._handler = logging.StreamHandler(sys.stderr)
        self._handler.setFormatter(formatter)

        # Set levels
        level_upper = level.upper()
        self._logger.setLevel(getattr(logging, level_upper))
        self._handler.setLevel(getattr(logging, level_upper))

        # Add handler to logger
        self._logger.addHandler(self._handler)

        # Prevent propagation to root logger to avoid duplicate messages
        self._logger.propagate = False

        self._current_level = level_upper

    def debug(self, message: str):
        """Log a debug message."""
        self._logger.debug(message)

    def info(self, message: str):
        """Log an info message."""
        self._logger.info(message)

    def warning(self, message: str):
        """Log a warning message."""
        self._logger.warning(message)

    def error(self, message: str):
        """Log an error message."""
        self._logger.error(message)

    def critical(self, message: str):
        """Log a critical message."""
        self._logger.critical(message)

    def trace(self, message: str):
        """Log a trace message (mapped to debug)."""
        self._logger.debug(message)


class FitsboltFormatter(logging.Formatter):
    """Custom formatter for fitsbolt logger with colors."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
        "BLUE": "\033[34m",  # Blue
        "GREEN": "\033[32m",  # Green
    }

    def format(self, record):
        # Format: time|fitsbolt-level| message
        time_str = self.formatTime(record, "%H:%M:%S")
        level_name = record.levelname
        message = record.getMessage()

        # Apply colors if stderr supports it
        if hasattr(sys.stderr, "isatty") and sys.stderr.isatty():
            colored_time = f"{self.COLORS['GREEN']}{time_str}{self.COLORS['RESET']}"
            colored_level = f"{self.COLORS['BLUE']}{level_name}{self.COLORS['RESET']}"

            # Color the message based on level
            level_color = self.COLORS.get(level_name, "")
            colored_message = f"{level_color}{message}{self.COLORS['RESET']}"

            return f"{colored_time}|fitsbolt-{colored_level}| {colored_message}"
        else:
            return f"{time_str}|fitsbolt-{level_name}| {message}"


# Create the module-level logger instance
logger = FitsboltLogger()
