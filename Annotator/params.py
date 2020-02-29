from __future__ import annotations

import re
import json
from dataclasses import dataclass, fields, field
from typing import Any, Optional, Union, List
from warnings import warn
import weakref

import numpy as np


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


class FRVertices(np.ndarray):
  connected = True

  def __new__(cls, inputArr: Union[list, np.ndarray]=None, connected=True, **kwargs):
    # See numpy docs on subclassing ndarray
    if inputArr is None:
      inputArr = np.zeros((0,2))
      if 'dtype' not in kwargs:
        kwargs['dtype'] = int
    arr = np.asarray(inputArr, **kwargs).view(cls)
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
      return self.reshape(-1)
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

class FRComplexVertices:
  """
  Unlike a simple list of vertices, :class:`FRComplexVertices` allows shapes with holes.
  It also knows how to serialize/deserialize itself for file storage.
  """
  globalVerts: List[FRVertices]
  """List of vertices in the component. Organized similar to opencv's contours."""

  hierarchy: np.ndarray
  """See hierarchy from opencv.findContours"""

  localPos: FRVertices
  """
  Offset into the global image. Useful for defining :class:FRComplexVertices
  within an ROI
  """


