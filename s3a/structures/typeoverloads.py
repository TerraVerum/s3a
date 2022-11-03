import typing as t
from pathlib import Path

import pandas as pd
from numpy import ndarray
from qtextras import OptionsDict
from qtextras.typeoverloads import FilePath

# `FilePath` Exposed for convenience

__all__ = [
    "BlackWhiteImg",
    "GrayImg",
    "RgbImg",
    "RgbaImg",
    "NChanImg",
    "OneDArr",
    "TwoDArr",
    "ThreeDArr",
    "LabelFieldType",
    "AnnParseError",
    "AnnInstanceError",
    "FilePath",
]

"""
Functions that just return a `np.ndarray` are often hard to interpret. Is the output a 
simple array, image, etc.? If it _is_ an image, what is the output shape? Developers 
and users will have to comb through the function body or rely on detailed documentation 
for an answer. This is a simple answer to the problem. By redefining np.ndarray in 
several ways, users and devs can more clearly interpret the intenionality behind 
various np.ndarray (and other) types. 
"""

BlackWhiteImg = ndarray
GrayImg = ndarray
RgbImg = ndarray
RgbaImg = ndarray
NChanImg = ndarray
OneDArr = ndarray
TwoDArr = ndarray
ThreeDArr = ndarray

LabelFieldType = t.Union[str, OptionsDict]


class AnnParseError(ValueError):
    def __init__(
        self,
        msg=None,
        file: Path = None,
        instances=None,
        invalidIndexes=None,
        **kwargs,
    ):
        self.fileName = file
        self.instances = instances
        self.invalidIndexes = invalidIndexes
        if msg is None and instances is not None and invalidIndexes is not None:
            msg = self.defaultErrorMsg()
        super().__init__(msg)

    def defaultErrorMsg(self):
        invalidInsts = self.instances[self.invalidIndexes]
        if isinstance(invalidInsts, pd.DataFrame):
            invalidInsts = invalidInsts.to_string()
        return (
            f"{self.fileName}: Encountered problems on annotation import:\n"
            f"{invalidInsts}"
        )


class AnnInstanceError(ValueError):
    pass
