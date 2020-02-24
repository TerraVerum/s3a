from __future__ import annotations

import re
from dataclasses import dataclass, fields, field
from typing import Any, Optional, Union
from warnings import warn
import weakref

import numpy as np
from pandas import DataFrame

# from Annotator.exceptions import FRParamParseError, FRIllFormedVertices


class FRParamParseError(Exception): pass
class FRIllFormedVertices(Exception): pass

@dataclass
class FRParam:
  name: str
  value: Any
  valType: Optional[Any] = None
  helpText: str = ''
  group: Optional[FRParamGroup] = None

  def __str__(self):
    return f'{self.name}'

  def __lt__(self, other):
    """
    Required for sorting by value in component table. Defer to alphabetic
    sorting
    :param other: Other :class:`FRParam` member for comparison
    :return: Whether `self` is less than `other`
    """
    return str(self) < str(other)

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
    # If we reach here the value didn't match any ComponentTypes values. Throw an error
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

  :param name: Display name of the parameter
  :param val: Initial value of the parameter. This is used within the program to infer
         parameter type, shape, comparison methods, etc.
  :param valType: Type of the variable if not easily inferrable from the value itself. For instance,
  class:`ShortcutParameter<Annotator.FRGraphics.parameditors.ShortcutParameter>` is indicated with string values
  (e.g. 'Ctrl+D'), so the user must explicitly specify that such an :class:`FRParam` is of type 'shortcut' (as
  defined in :class:`ShortcutParameter<Annotator.FRGraphics.parameditors.ShortcutParameter>`) If the type *is* easily
  inferrable, this may be left blank.
  :param helpText: Additional documentation for this parameter.
  :return: Field that can be inserted within the :class:`FRParamGroup` dataclass.
  """
  if valType is None:
    # Infer from value
    # TODO: Is there an easier way?
    # The string representation of val 'type' is: <class 'type'>
    # Parse for '' occurrences and grab in between the quotes
    valType = re.search('\'.*\'', str(type(val))).group()[1:-1]
  return field(default_factory=lambda: FRParam(name, val, valType, helpText))


# class FRVertices(DataFrame):
#
#   _metadata = ['connected']
#
#   def __init__(self, *args, **kwargs):
#     """
#     Overload pandas dataframe and only allow columns conducive for storing shape
#     vertex points
#     """
#     self.connected = kwargs.pop('connected', True)
#     columns = {'columns': ['x', 'y']}
#     kwargs.update(columns)
#     super().__init__(*args, **kwargs)
#
#   @property
#   def _constructor(self):
#     """
#     This black magic is required for pandas subclassing so that dataframes retain
#     subclass information after math operations
#     """
#     def f(*args, **kwargs):
#       tmp = DataFrame(*args, **kwargs)
#       df = FRVertices(tmp.values).__finalize__(self)
#       return df
#
#     return f
#
#   def asPoint(self) -> np.ndarray:
#     """
#     Treats the current FRVertices object as if it only contained one vertex that can
#     be treated as a point. If that condition holds, this point is returned as a
#     numpy array (x,y)
#     :return: Numpy point (x,y)
#     """
#     if len(self) > 1:
#       raise FRIllFormedVertices(f'Cannot call asPoint on vertex list containing'
#                                 f' more than one row. List currently contains {len(self)}'
#                                 f' points.')
#     return self.to_numpy().flatten()
#
#   def astype(self, *args, **kwargs):
#     """
#     Preserve type information when type casting
#     """
#     return FRVertices(super().astype(*args, **kwargs))
#
#   @property
#   def rows(self): return self.y
#
#   @property
#   def cols(self): return self.x

class FRVertices(np.ndarray):
  connected = True
  def __new__(cls, inputArr: Union[list, np.ndarray]=None, connected=True, **kwargs):
    if inputArr is None:
      inputArr = np.zeros((0,2))
    arr = np.asarray(inputArr).view(cls)
    arr.connected = connected
    return arr

  def __array_finalize__(self, obj):
    shape = self.shape
    shapeLen = len(shape)
    # indicates point, so the one dimension must have only 2 elements
    if 1 < shapeLen < 2 and shape[0] != 2:
      raise FRIllFormedVertices(f'A one-dimensional vertex array must be shape (2,).'
                                f' Receieved array of shape {shape}')
    elif shapeLen > 2 or shapeLen > 1 and shape[1] != 2:
      raise FRIllFormedVertices(f'Vertex list must be Nx2. Received shape {shape}.')
    if obj is None: return
    self.connected = getattr(obj, 'connected', True)

  def asPoint(self):
    if self.size == 2:
      return self.flatten()
    # Reaching here means the user requested vertices as point when
    # more than one point is in the list
    raise FRIllFormedVertices(f'asPoint() can only be called when one vertex is in'
                              f' the vertex list. Currently has shape {self.shape}')

  def nonNanEntries(self):
    return self[~np.isnan(self[:,0])]


  @property
  def x(self):
    # Copy to array first so dimensionality checks are no longer required
    return np.array(self).reshape(-1,2)[:,[0]]
  @x.setter
  def x(self, newX): self.reshape(-1,2)[:,0] = newX

  @property
  def y(self):
    return np.array(self).reshape(-1,2)[:,[1]]
  @y.setter
  def y(self, newY): self.reshape(-1,2)[:,1] = newY

  @property
  def rows(self): return self.y
  @rows.setter
  def rows(self, newRows): self.y = newRows

  @property
  def cols(self):return self.x
  @cols.setter
  def cols(self, newCols):self.x = newCols

