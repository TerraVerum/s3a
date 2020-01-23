from dataclasses import dataclass, fields, field
from typing import Any


@dataclass
class ABParam:
  name: str
  value: Any

  def __str__(self):
    return f'{self.name}'

  def __hash__(self):
    # Since every param within a group will have a unique name, just the name is
    # sufficient to form a proper hash
    return hash(self.name,)

@dataclass
class ABParamGroup:
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

def newParam(name, val=None):
  """
  Factory for creating new parameters within a :class:`ABParamGroup`.

  :param name: Display name of the parameter
  :param val: Initial value of the parameter. This is used within the program to infer
         parameter type, shape, comparison methods, etc.
  :return: Field that can be inserted within the :class:`ABParamGroup` dataclass.
  """
  return field(default_factory=lambda: ABParam(name, val))