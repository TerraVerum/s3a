from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, field
from functools import partial
from typing import Any, Optional, Union
from warnings import warn
import weakref

import numpy as np

from .exceptions import FRParamParseError, FRIllFormedVertices

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

  connected: bool = True

  @staticmethod
  def __new__(cls, shape=None, *args, **kwargs):
    # Make sure the shape is appropriate for vertices
    if shape is None:
      # Default to empty Nx2 array
      shape = (0, 2)
    elif len(shape) != 2 or shape[1] != 2:
      raise FRIllFormedVertices(f"Vertices for FRVertices must be Nx2.\n"
                                f"Received shape was {shape}")
    return super().__new__(FRVertices, shape, *args, **kwargs)

  @staticmethod
  def createFromArr(npArrToCopy: Union[np.ndarray, list]) -> FRVertices:
    """
    Construct a FRVertices object from the given numpy array (copy constructor).
    Using the __init__ of this class instead of directly copy()ing the data ensures the
    shape of :param:`npArrToCopy` is correct.

    :param npArrToCopy:
    :return: FRVertices object
    """
    if isinstance(npArrToCopy, list):
      npArrToCopy = np.array(npArrToCopy)
    outObj = FRVertices(npArrToCopy.shape)
    outObj[:] = npArrToCopy
    return outObj

  @property
  def x(self): return self[:,[0]]
  @x.setter
  def x(self, newX): self[:,0] = newX

  @property
  def y(self): return self[:, [1]]
  @y.setter
  def y(self, newY): self[:, 1] = newY

  @property
  def rows(self): return self.y
  @rows.setter
  def rows(self, newY): self.y = newY

  @property
  def cols(self): return self.x
  @cols.setter
  def cols(self, newX): self.x = newX

@dataclass
class FRDrawShape:
  type: FRParam
  points: FRVertices = field(default_factory=FRVertices)

class FREditablePropFunc(ABC):
  sharedProps: FRParamGroup

  @abstractmethod
  def __call__(self, *args, **kwargs):
    pass

class FRImageProcessor(ABC):
  image: np.ndarray


  def localCompEstimate(self, prevCompMask: np.ndarray, drawShape: FRDrawShape) -> np.ndarray:
    pass


  def globalCompEstimate(self) -> np.ndarray:
    pass


if __name__ == '__main__':
    x = np.ones((5,2))
    t = FRVertices.createFromArr(x)
    print(t)