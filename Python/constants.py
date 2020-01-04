from enum import Enum
from pathlib import Path
import os

# Preference directories
BASE_DIR = os.path.dirname(Path(__file__).absolute())
SCHEMES_DIR = os.path.join(BASE_DIR, 'Schemes', '')
LAYOUTS_DIR = os.path.join(BASE_DIR, 'Layouts', '')


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
    return f'Type: {self.value}'

  def __lt__(self, other):
    """
    Required for sorting by enum value in component table. Defer to alphabetic
    sorting
    :param other: Other :class:`ComponentTypes` member for comparison
    :return: Whether `self` is less than `other`
    """
    return str(self) < str(other)

# noinspection PyPep8
class ComponentTableFields(Enum):
  INST_ID: Enum = 'Instance ID'
  VALIDATED: Enum = 'Validated'
  DEVICE_TYPE: Enum = 'Device Type'
  LOGO: Enum = 'Logo'
  NOTES: Enum = 'Notes'
  BOARD_TEXT: Enum = 'Board Text'
  DEVICE_TEXT: Enum = 'Device Text'
  VERTICES: Enum = 'Vertices'

# Define shorthand terms for convenience
_ = ComponentTableFields
TABLE_EDIT_DICT = [
  {'name': _.VALIDATED.value, 'type': 'bool', 'value': False},
  {'name': _.DEVICE_TYPE.value, 'type': 'list', 'value': ComponentTypes.N_A.value,
   'limits': [compType for compType in ComponentTypes]
   },
  {'name': _.LOGO.value, 'type': 'text', 'value': ''},
  {'name': _.NOTES.value, 'type': 'text', 'value': ''},
  {'name': _.BOARD_TEXT.value, 'type': 'text', 'value': ''},
  {'name': _.DEVICE_TEXT.value, 'type': 'text', 'value': ''}
]


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


# Define shorthand terms for convenience
_ = SchemeValues
DEFAULT_SCHEME_DICT = [
  {'name': _.COMP_PARAMS.value, 'type': 'group', 'children': [
    {'name': _.VALID_ID_COLOR.value, 'type': 'color', 'value': '0f0'},
    {'name': _.NONVALID_ID_COLOR.value, 'type': 'color', 'value': 'f00'},
    {'name': _.BOUNDARY_COLOR.value, 'type': 'color', 'value': 'ff0'},
    {'name': _.BOUNDARY_WIDTH.value, 'type': 'int', 'value': 2},
    {'name': _.ID_FONT_SIZE.value, 'type': 'int', 'value': 10}
  ]},
  {'name': _.FOC_IMG_PARAMS.value, 'type': 'group', 'children': [
    {'name': _.REG_VERT_COLOR.value, 'type': 'color', 'value': '0f0'},
    {'name': _.REG_FILL_COLOR.value, 'type': 'color', 'value': '00ff0046'}
  ]},
]
