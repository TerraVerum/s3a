from enum import Enum


class DockNames(Enum):
  MAIN_CTRL = 'Image Import Controls'
  MAIN_IMG = 'Main Image'
  COMP_CTRL = 'Component Controls'
  COMP_IMG = 'Component Image'
  SPREAD_CTRL = 'Spreadsheet Controls'

class ComponentTypes(Enum):
  CAP = 'Capacitor'
  RES = 'Resistor'
  IND = 'Inductor'
  IC = 'IC'
  N_A = 'Unassigned'