from enum import Enum

class ComponentTypes(Enum):
  CAP:Enum = 'Capacitor'
  RES:Enum = 'Resistor'
  IND:Enum = 'Inductor'
  IC:Enum = 'IC'
  N_A:Enum = 'Unassigned'

class SchemeValues(Enum):
  COMP_PARAMS:Enum = 'Component Parameters'
  validIdColor:Enum = 'Validated ID Color'
  nonValidIdColor:Enum = 'Non-Validated ID Color'
  boundaryColor:Enum = 'Component Boundary Color'
  boundaryWidth:Enum = 'Component Boundary Width'
  idFontSize:Enum = 'ID Font Size'

  FOC_IMG_PARAMS:Enum = 'Focused Image Parameters'
  foc_vertColor:Enum = 'Vertex Color'
  foc_fillColor:Enum = 'Fill Color'

# Define shorthand terms for convenience
_ = SchemeValues
DEFAULT_SCHEME_DICT = [
  {'name': _.COMP_PARAMS.value, 'type': 'group', 'children': [
    {'name': _.validIdColor.value, 'type': 'color', 'value': '0f0'},
    {'name': _.nonValidIdColor.value, 'type': 'color', 'value': 'f00'},
    {'name': _.boundaryColor.value, 'type': 'color', 'value': 'ff0'},
    {'name': _.boundaryWidth.value, 'type': 'int', 'value': 2},
    {'name': _.idFontSize.value, 'type': 'int', 'value': 10}
  ]},
  {'name': _.FOC_IMG_PARAMS.value, 'type': 'group', 'children': [
    {'name': _.foc_vertColor.value, 'type': 'color', 'value': '0f0'},
    {'name': _.foc_fillColor.value, 'type': 'color', 'value': '00ff0046'}
  ]},
]