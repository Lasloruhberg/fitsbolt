# fitsbolt - A Python package for image loading and processing
# Copyright (C) <2025>  <Ruhberg>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the MIT License

import sys
import logging
from typing import Optional


def _is_jupyter_notebook():
    """Detect if code is running in a Jupyter notebook environment."""
    try:
        # Check for IPython kernel
        from IPython import get_ipython

        ipython = get_ipython()
        if ipython is None:
            return False

        # Check if it's a notebook kernel (not terminal IPython)
        return hasattr(ipython, "kernel")
    except ImportError:
        return False


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
    """Custom formatter for fitsbolt logger with colors for both terminal and Jupyter."""

    # ANSI color codes for terminal
    ANSI_COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
        "BLUE": "\033[34m",  # Blue
        "GREEN": "\033[32m",  # Green
    }

    # HTML colors for Jupyter notebooks
    HTML_COLORS = {
        "DEBUG": "#17A2B8",  # Cyan
        "INFO": "#28A745",  # Green
        "WARNING": "#FFC107",  # Yellow
        "ERROR": "#DC3545",  # Red
        "CRITICAL": "#6F42C1",  # Purple
        "BLUE": "#007BFF",  # Blue
        "GREEN": "#28A745",  # Green
        "TIME": "#28A745",  # Green for timestamp
    }

    def __init__(self):
        super().__init__()
        self.is_jupyter = _is_jupyter_notebook()

    def format(self, record):
        # Format: time|fitsbolt-level| message
        time_str = self.formatTime(record, "%H:%M:%S")
        level_name = record.levelname
        message = record.getMessage()

        if self.is_jupyter:
            # Use HTML formatting for Jupyter notebooks
            return self._format_html(time_str, level_name, message)
        else:
            # Use ANSI colors for terminal
            return self._format_ansi(time_str, level_name, message)

    def _format_html(self, time_str, level_name, message):
        """Format log message with HTML for Jupyter notebooks."""
        try:
            from IPython.display import HTML, display

            time_color = self.HTML_COLORS["TIME"]
            level_color = self.HTML_COLORS["BLUE"]
            message_color = self.HTML_COLORS.get(level_name, "#000000")

            html_content = (
                f'<span style="color: {time_color}; font-weight: bold;">{time_str}</span>'
                f"|fitsbolt-"
                f'<span style="color: {level_color}; font-weight: bold;">{level_name}</span>'
                f"| "
                f'<span style="color: {message_color};">{message}</span>'
            )

            # Display the HTML directly in Jupyter
            display(HTML(html_content))

            # Return plain text for logging system
            return f"{time_str}|fitsbolt-{level_name}| {message}"
        except ImportError:
            # Fallback to plain text if IPython is not available
            return f"{time_str}|fitsbolt-{level_name}| {message}"

    def _format_ansi(self, time_str, level_name, message):
        """Format log message with ANSI colors for terminal."""
        # Apply colors if stderr supports it
        if hasattr(sys.stderr, "isatty") and sys.stderr.isatty():
            colored_time = f"{self.ANSI_COLORS['GREEN']}{time_str}{self.ANSI_COLORS['RESET']}"
            colored_level = f"{self.ANSI_COLORS['BLUE']}{level_name}{self.ANSI_COLORS['RESET']}"

            # Color the message based on level
            level_color = self.ANSI_COLORS.get(level_name, "")
            colored_message = f"{level_color}{message}{self.ANSI_COLORS['RESET']}"

            return f"{colored_time}|fitsbolt-{colored_level}| {colored_message}"
        else:
            return f"{time_str}|fitsbolt-{level_name}| {message}"


# Create the module-level logger instance
logger = FitsboltLogger()
