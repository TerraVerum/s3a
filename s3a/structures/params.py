from __future__ import annotations

import weakref
from typing import Optional, Collection, Union
from warnings import warn

from utilitys import PrjParam

__all__ = ["PrjParam", "PrjParamGroup"]


class PrjParamGroup:
    """
    Hosts all child parameters and offers convenience function for iterating over them
    """

    def __init__(self, fields=None):
        self.fields = fields if len(fields) else []

    def paramNames(self):
        """
        Outputs the column names of each parameter in the group.
        """
        return [curField.name for curField in self]

    def __iter__(self):
        yield from self.fields

    def __len__(self):
        return len(self.fields)

    def __str__(self):
        return f", ".join([f.name for f in self])

    def __post_init__(self):
        for param in self:
            param.group = weakref.proxy(self)

    @staticmethod
    def fieldFromParam(
        group: Collection[PrjParam],
        param: Union[str, PrjParam],
        default: PrjParam = None,
    ):
        """
        Allows user to create a :class:`PrjParam` object from its string value (or a parameter that
        can equal one of the parameters in this list)
        """
        param = str(param).lower()
        for matchParam in group:
            if str(matchParam).lower() == param:
                return matchParam
        # If we reach here the value didn't match any CNSTomponentTypes values. Throw an error
        if default is None and hasattr(group, "getDefault"):
            default = group.getDefault()
        baseWarnMsg = f'String representation "{param}" was not recognized.\n'
        if default is None:
            # No default specified, so we have to raise Exception
            raise ValueError(
                baseWarnMsg
                + f'Must be one of {", ".join(list(str(g) for g in group))}.'
            )
        # No exception needed, since the user specified a default type in the derived class
        warn(baseWarnMsg + f"Defaulting to {default}", UserWarning)
        return default

    @classmethod
    def getDefault(cls) -> Optional[PrjParam]:
        """
        Returns the default Param from the group. This can be overloaded in derived classes to yield a safe
        fallback class if the :func:`fieldFromParam` method fails.
        """
        return None
