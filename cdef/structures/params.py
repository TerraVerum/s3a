from __future__ import annotations
import re
import weakref
from dataclasses import dataclass, fields, field
from typing import Any, Optional
from typing_extensions import Protocol, runtime_checkable
from warnings import warn

from .exceptions import FRParamParseError

@runtime_checkable
class ContainsSharedProps(Protocol):
  @classmethod
  def __initEditorParams__(cls):
    return


@dataclass
class FRParam:
  name: str
  """Display name of the parameter"""
  value: Any = None
  """
  Initial value of the parameter.
    This is used within the program to infer parameter type, shape, comparison methods, etc.
  """
  valType: Optional[str] = None
  """
  Type of the variable if not easily inferrable from the value itself. 
    For instance, class:`FRShortcutParameter<cdef.frgraphics.parameditors.FRShortcutParameter>`
    is indicated with string values (e.g. 'Ctrl+D'), so the user must explicitly specify 
    that such an :class:`FRParam` is of type 'shortcut' (as defined in 
    :class:`FRShortcutParameter<cdef.frgraphics.parameditors.FRShortcutParameter>`)
    If the type *is* easily inferrable, this may be left blank.
  """
  helpText: str = ''
  """Additional documentation for this parameter."""
  group: Optional[FRParamGroup] = None
  """FRParamGroup to which this parameter belongs, if this parameter is part of
    a group. This is set by the FRParamGroup, not manually
  """

  def __post_init__(self):
    if self.valType is None:
      # Infer from value
      valType = type(self.value).__name__
      self.valType = valType

  def __str__(self):
    return f'{self.name}'

  def __repr__(self):
    return f'FRParam(name=\'{self.name}\', value=\'{self.value}\', ' \
           f'valType=\'{self.valType}\', helpText=\'{self.helpText}\', ' \
           f'group=FRParamGroup(...))'

  def __lt__(self, other):
    """
    Required for sorting by value in component table. Defer to alphabetic
    sorting
    :param other: Other :class:`FRParam` member for comparison
    :return: Whether `self` is less than `other`
    """
    return str(self) < str(other)

  def __eq__(self, other):
    # TODO: Highly naive implementation. Be sure to make this more robust if it needs to be
    #   for now assume only other frparams will be passed in
    return repr(self) == repr(other)

  def __hash__(self):
    # Since every param within a group will have a unique name, just the name is
    # sufficient to form a proper hash
    return hash(self.name,)


@dataclass
class FRParamGroup:
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

  def __post_init__(self):
    for param in self:
      param.group = weakref.proxy(self)

  def fromString(self, paramName: str):
    """
    Allows user to create a :class:`FRParam` object from its string value
    """
    paramName = paramName.lower()
    for param in self:
      if param.name.lower() == paramName:
        return param
    # If we reach here the value didn't match any FRComponentTypes values. Throw an error
    defaultParam = self.getDefault()
    baseWarnMsg = f'String representation "{paramName}" was not recognized. '
    if defaultParam is None:
      # No default specified, so we have to raise Exception
      raise FRParamParseError(baseWarnMsg + 'No class default is specified.')
    # No exception needed, since the user specified a default type in the derived class
    warn(baseWarnMsg + f'Defaulting to {defaultParam.name}')
    return defaultParam

  @classmethod
  def getDefault(cls) -> Optional[FRParam]:
    """
    Returns the default Param from the group. This can be overloaded in derived classes to yield a safe
    fallback class if the :func:`fromString` method fails.
    """
    return None


def newParam(name, val=None, valType=None, helpText=''):
  """
  Factory for creating new parameters within a :class:`FRParamGroup`.

  See parameter documentation from :class:FRParam for arguments.

  :return: Field that can be inserted within the :class:`FRParamGroup` dataclass.
  """
  return field(default_factory=lambda: FRParam(name, val, valType, helpText))