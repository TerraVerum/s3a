from dataclasses import dataclass, fields, field
from typing import Any
from enum import Enum
from abc import ABC

from pathlib import Path
import os

from pandas import DataFrame as df
import numpy as np

# Preference directories
BASE_DIR = os.path.dirname(Path(__file__).absolute())
SCHEMES_DIR = os.path.join(BASE_DIR, 'Schemes', '')
LAYOUTS_DIR = os.path.join(BASE_DIR, 'Layouts', '')
FILTERS_DIR = os.path.join(BASE_DIR, 'Filters', '')
REGION_CTRL_DIR = os.path.join(BASE_DIR, 'RegionControls', '')


class ComponentTypes(Enum):
  CAP: Enum = 'Capacitor'
  RES: Enum = 'Resistor'
  IND: Enum = 'Inductor'
  IC: Enum = 'IC'
  N_A: Enum = 'Unassigned'

  def __str__(self):
    """
    Redefine print representation for prettier display in table
    :return: printable string representation of enum object
    """
    return f'{self.value}'

  def __lt__(self, other):
    """
    Required for sorting by enum value in component table. Defer to alphabetic
    sorting
    :param other: Other :class:`ComponentTypes` member for comparison
    :return: Whether `self` is less than `other`
    """
    return str(self) < str(other)


@dataclass()
class ABParam:
  name: str
  value: Any

  def __str__(self):
    return f'{self.name}: {self.value}'

class ABParamGroup():
  @classmethod
  def paramNames(cls):
    """
    Outputs the column names of each parameter in the group. Helpful since it
    doesn't require a class instantiation.
    """
    paramNames = []
    for field in fields(cls):
      paramNames.append(field.default.name)
    return paramNames

  def __iter__(self):
    for field in fields(self):
      yield getattr(self, field.name)

  def to_dataframe(self):
    df_list = []
    df_colNames = []
    for param in self:
      df_colNames.append(param.name)
      df_list.append(param.value)
    return df([df_list], columns=df_colNames)


_vertDefault = lambda: ABParam('Vertices', np.ones((1,2))*np.nan)
@dataclass
class NewComponentTableFields(ABParamGroup):
  INST_ID:ABParam    = ABParam('Instance ID', -1)
  VALIDATED:ABParam  = ABParam('Validated', False)
  DEV_TYPE:ABParam   = ABParam('Device Type', ComponentTypes.N_A)
  DEV_TEXT:ABParam   = ABParam('Device Text', '')
  BOARD_TEXT:ABParam = ABParam('Board Text', '')
  LOGO:ABParam       = ABParam('Logo', '')
  NOTES:ABParam      = ABParam('Notes', '')
  VERTICES:ABParam   = field(default_factory=_vertDefault)

class ComponentTableFields(Enum):
  INST_ID: Enum = 'Instance ID'
  VALIDATED: Enum = 'Validated'
  DEVICE_TYPE: Enum = 'Device Type'
  LOGO: Enum = 'Logo'
  NOTES: Enum = 'Notes'
  BOARD_TEXT: Enum = 'Board Text'
  DEVICE_TEXT: Enum = 'Device Text'
  VERTICES: Enum = 'Vertices'


class SchemeValues(Enum):
  COMP_PARAMS: Enum = 'Component Parameters'
  VALID_ID_COLOR: Enum = 'Validated ID Color'
  NONVALID_ID_COLOR: Enum = 'Non-Validated ID Color'
  BOUNDARY_COLOR: Enum = 'Component Boundary Color'
  BOUNDARY_WIDTH: Enum = 'Component Boundary Width'
  ID_FONT_SIZE: Enum = 'ID Font Size'

  FOC_IMG_PARAMS: Enum = 'Focused Image Parameters'
  REG_VERT_COLOR: Enum = 'Vertex Color'
  REG_FILL_COLOR: Enum = 'Fill Color'

class RegionControlsEditorValues(Enum):
  MARGIN: Enum = 'Margin'
  SEG_THRESH: Enum = 'Segmentation Threshold'
  SEED_THRESH: Enum = 'Seedpoint Mean Threshold'
  NEW_COMP_SZ: Enum = 'New Component Size'

if __name__ == '__main__':
  x1 = NewComponentTableFields()
  x2 = NewComponentTableFields()
  x2.VERTICES.value[0,1] = 3
  print(x1.VERTICES)