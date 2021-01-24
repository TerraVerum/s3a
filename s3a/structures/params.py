from __future__ import annotations

import weakref
from dataclasses import dataclass, fields, field
from typing import Any, Optional, Collection, Union
from warnings import warn

from utilitys import PrjParam

@dataclass
class PrjParamGroup:
  """
  Hosts all child parameters and offers convenience function for iterating over them
  """

  def paramNames(self):
    """
    Outputs the column names of each parameter in the group.
    """
    return [curField.name for curField in self]

  def __iter__(self):
    # 'self' is an instance of the class, so the warning is a false positive
    # noinspection PyDataclass
    for curField in fields(self):
      yield getattr(self, curField.name)

  def __len__(self):
    return len(fields(self))

  def __str__(self):
    return f'{[f.name for f in self]}'

  def __post_init__(self):
    for param in self:
      param.group = weakref.proxy(self)

  @staticmethod
  def fieldFromParam(group: Union[Collection[PrjParam], PrjParamGroup], param: Union[str, PrjParam],
                     default: PrjParam=None):
    """
    Allows user to create a :class:`PrjParam` object from its string value (or a parameter that
    can equal one of the parameters in this list)
    """
    param = str(param).lower()
    for matchParam in group:
      if matchParam.name.lower() == param:
        return matchParam
    # If we reach here the value didn't match any CNSTomponentTypes values. Throw an error
    if default is None and hasattr(group, 'getDefault'):
      default = group.getDefault()
    baseWarnMsg = f'String representation "{param}" was not recognized.\n'
    if default is None:
      # No default specified, so we have to raise Exception
      raise ValueError(baseWarnMsg + 'No class default is specified.')
    # No exception needed, since the user specified a default type in the derived class
    warn(baseWarnMsg + f'Defaulting to {default.name}', UserWarning)
    return default

  @classmethod
  def getDefault(cls) -> Optional[PrjParam]:
    """
    Returns the default Param from the group. This can be overloaded in derived classes to yield a safe
    fallback class if the :func:`fieldFromParam` method fails.
    """
    return None

def newParam(name: str, val: Any=None, pType: str=None, helpText='', **opts):
  """
  Factory for creating new parameters within a :class:`PrjParamGroup`.

  See parameter documentation from :class:PrjParam for arguments.

  :return: Field that can be inserted within the :class:`PrjParamGroup` dataclass.
  """
  return field(default_factory=lambda: PrjParam(name, val, pType, helpText, **opts))