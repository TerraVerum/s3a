from __future__ import annotations

import weakref
from typing import Collection, Optional, Union
from warnings import warn

from qtextras import OptionsDict

__all__ = ["OptionsDict", "OptionsDictGroup"]


class OptionsDictGroup:
    """
    Hosts all child parameters and offers convenience function for iterating over them
    """

    def __init__(self, fields=None):
        self.fields = fields if len(fields) else []

    def parameterNames(self):
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
    def fieldFromParameter(
        group: Collection[OptionsDict],
        parameter: Union[str, OptionsDict],
        default: OptionsDict = None,
    ):
        """
        Allows user to create a :class:`OptionsDict` object from its string value (or a
        parameter that can equal one of the parameters in this list)
        """
        parameter = str(parameter).lower()
        for matchParam in group:
            if str(matchParam).lower() == parameter:
                return matchParam
        # If we reach here the value didn't match any CNSTomponentTypes values. Throw
        # an error
        if default is None and hasattr(group, "getDefault"):
            default = group.getDefault()
        baseWarnMsg = f'String representation "{parameter}" was not recognized.\n'
        if default is None:
            # No default specified, so we have to raise Exception
            raise ValueError(
                baseWarnMsg
                + f'Must be one of {", ".join(list(str(g) for g in group))}.'
            )
        # No exception needed, since the user specified a default type in the derived
        # class
        warn(baseWarnMsg + f"Defaulting to {default}", UserWarning, stacklevel=2)
        return default

    @classmethod
    def getDefault(cls) -> Optional[OptionsDict]:
        """
        Returns the default Param from the group. This can be overloaded in derived
        classes to yield a safe fallback class if the :func:`fieldFromParam` method
        fails.
        """
        return None
