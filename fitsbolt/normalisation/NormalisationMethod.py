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
