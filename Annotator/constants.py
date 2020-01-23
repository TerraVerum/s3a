import os
from dataclasses import dataclass, fields, field
from enum import Enum
from pathlib import Path
from typing import Any
from warnings import warn

import numpy as np

# Preference directories
BASE_DIR = os.path.dirname(Path(__file__).absolute())
SCHEMES_DIR     = os.path.join(BASE_DIR, 'MenuOpts', 'Schemes', '')
LAYOUTS_DIR     = os.path.join(BASE_DIR, 'MenuOpts', 'Layouts', '')
FILTERS_DIR     = os.path.join(BASE_DIR, 'MenuOpts', 'Filters', '')
REGION_CTRL_DIR = os.path.join(BASE_DIR, 'MenuOpts', 'RegionControls', '')
# Ensure these directories exist
for curDir in [SCHEMES_DIR, LAYOUTS_DIR, FILTERS_DIR, REGION_CTRL_DIR]:
  Path(curDir).mkdir(parents=True, exist_ok=True)


class ComponentTypes(Enum):
  CAP: Enum = 'Capacitor'
  RES: Enum = 'Resistor'
  IND: Enum = 'Inductor'
  IC: Enum = 'IC'
  N_A: Enum = 'Unassigned'

  @classmethod
  def fromString(cls, value):
    """
    Allows user to create a ComponentTypes object from its string value
    """
    value = value.lower()
    for param in cls:
      if param.value.lower() == value:
        return param
    # If we reach here the value didn't match any ComponentTypes values. Throw an error
    warn('String representation of ComponentTypes was not recognized. Defaulting to Unassigned')
    return cls.N_A

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

newParam = lambda name, val=None: field(default_factory=lambda:ABParam(name, val))
@dataclass
class CompParams(ABParamGroup):
  # These 3 params MUST exist in the component
  INST_ID   :ABParam  = newParam('Instance ID', -1)
  VERTICES  :ABParam  = newParam('Vertices', np.ones((1,2))*np.nan)
  VALIDATED :ABParam  = newParam('Validated', False)

  DEV_TEXT   :ABParam = newParam('Device Text', '')
  DEV_TYPE   :ABParam = newParam('Device Type', ComponentTypes.N_A)
  BOARD_TEXT :ABParam = newParam('Board Text', '')
  LOGO       :ABParam = newParam('Logo', '')
  NOTES      :ABParam = newParam('Notes', '')
TEMPLATE_COMP = CompParams()

@dataclass
class SchemeValues(ABParamGroup):
  COMP_PARAMS       :ABParam = newParam('Component Parameters')
  VALID_ID_COLOR    :ABParam = newParam('Validated ID Color')
  NONVALID_ID_COLOR :ABParam = newParam('Non-Validated ID Color')
  BOUNDARY_COLOR    :ABParam = newParam('Component Boundary Color')
  BOUNDARY_WIDTH    :ABParam = newParam('Component Boundary Width')
  ID_FONT_SIZE      :ABParam = newParam('ID Font Size')

  FOC_IMG_PARAMS    :ABParam = newParam('Focused Image Parameters')
  REG_VERT_COLOR    :ABParam = newParam('Vertex Color')
  REG_FILL_COLOR    :ABParam = newParam('Fill Color')
TEMPLATE_SCHEME_VALUES = SchemeValues()

@dataclass
class RegionControlsEditorValues(ABParamGroup):
  MARGIN              :ABParam = newParam('Margin')
  SEG_THRESH          :ABParam = newParam('Segmentation Threshold')
  SEED_THRESH         :ABParam = newParam('Seedpoint Mean Threshold')
  NEW_COMP_SZ         :ABParam = newParam('New Component Size')
  EST_BOUNDS_ON_START :ABParam = newParam('Estimate Boundaries on Image Load')
TEMPLATE_REG_CTRLS = RegionControlsEditorValues()

if __name__ == '__main__':
  c = ComponentTypes.fromString('capacitor')
  print(c)