# -*- coding: utf-8 -*-
"""Python 3.10 compatibility shims for running the test suite.

The package uses `typing.Self` and `enum.StrEnum`, which were added in
Python 3.11, while `pyproject.toml` declares support for Python >= 3.10.
"""

import enum
import sys
import typing

if sys.version_info < (3, 11):
    if not hasattr(typing, "Self"):
        import typing_extensions

        typing.Self = typing_extensions.Self
    if not hasattr(enum, "StrEnum"):

        class StrEnum(str, enum.Enum):
            def __str__(self) -> str:
                return str(self.value)

        enum.StrEnum = StrEnum
