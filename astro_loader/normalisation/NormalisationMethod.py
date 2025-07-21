from enum import IntEnum


class NormalisationMethod(IntEnum):
    """Enum handling different normalisation methods."""
    CONVERSION_ONLY = 0
    LOG = 1
    ZSCALE = 2
    ASINH = 3

    @classmethod
    def get_options(cls):
        """Returns a list of tuples (label, value)"""
        return [
            ("ConversionOnly", cls.CONVERSION_ONLY),
            ("LogStretch", cls.LOG),
            ("ZscaleInterval", cls.ZSCALE),
            ("Asinh", cls.ASINH),
        ]

    @classmethod
    def get_test_methods(cls):
        """Returns all methods for testing purposes."""
        return [cls.CONVERSION_ONLY, cls.LOG, cls.ZSCALE, cls.ASINH]
