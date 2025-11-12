# fitsbolt - A Python package for image loading and processing
# Copyright (C) <2025>  <Ruhberg>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the MIT License

from enum import IntEnum


class NormalisationMethod(IntEnum):
    """Enum handling different normalisation methods."""

    CONVERSION_ONLY = 0
    LOG = 1
    ZSCALE = 2
    ASINH = 3
    LINEAR = 4
    MIDTONES = 5

    @classmethod
    def get_options(cls):
        """Returns a list of tuples (label, value)"""
        return [
            ("ConversionOnly", cls.CONVERSION_ONLY),
            ("LogStretch", cls.LOG),
            ("ZscaleInterval", cls.ZSCALE),
            ("Asinh", cls.ASINH),
            ("Linear", cls.LINEAR),
            ("Midtones", cls.MIDTONES),
        ]

    @classmethod
    def get_test_methods(cls):
        """Returns all methods for testing purposes."""
        return [cls.CONVERSION_ONLY, cls.LOG, cls.ZSCALE, cls.ASINH, cls.LINEAR, cls.MIDTONES]
