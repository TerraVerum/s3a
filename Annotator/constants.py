import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

# Preference directories
from Annotator.params import ABParam, ABParamGroup, newParam

BASE_DIR = os.path.dirname(Path(__file__).absolute())
SCHEMES_DIR = os.path.join(BASE_DIR, 'MenuOpts', 'Schemes', '')
LAYOUTS_DIR = os.path.join(BASE_DIR, 'MenuOpts', 'Layouts', '')
FILTERS_DIR = os.path.join(BASE_DIR, 'MenuOpts', 'Filters', '')
GEN_PROPS_DIR = os.path.join(BASE_DIR, 'MenuOpts', 'RegionControls', '')
SHORTCUTS_DIR = os.path.join(BASE_DIR, 'MenuOpts', 'Shortcuts', '')

# Ensure these directories exist
for curDir in [SCHEMES_DIR, LAYOUTS_DIR, FILTERS_DIR,
               GEN_PROPS_DIR, SHORTCUTS_DIR]:
  Path(curDir).mkdir(parents=True, exist_ok=True)


@dataclass
class ComponentTypes(ABParamGroup):
  CAP   : ABParam = newParam('Capacitor')
  RES   : ABParam = newParam('Resistor')
  IND   : ABParam = newParam('Inductor')
  TRANS : ABParam = newParam('Transistor')
  IC    : ABParam = newParam('IC')
  OTHER : ABParam = newParam('Other')
  N_A   : ABParam = newParam('Unassigned')

TEMPLATE_COMP_TYPES = ComponentTypes()


@dataclass
class CompParams(ABParamGroup):
  # These 3 params MUST exist in the component
  INST_ID    : ABParam = newParam('Instance ID', -1)
  VERTICES   : ABParam = newParam('Vertices', np.ones((1, 2)) * np.nan)
  VALIDATED  : ABParam = newParam('Validated', False)

  DEV_TEXT   : ABParam = newParam('Device Text', '')
  DEV_TYPE   : ABParam = newParam('Device Type', TEMPLATE_COMP_TYPES.N_A)
  BOARD_TEXT : ABParam = newParam('Board Text', '')
  LOGO       : ABParam = newParam('Logo', '')
  NOTES      : ABParam = newParam('Notes', '')

TEMPLATE_COMP = CompParams()


@dataclass
class _ABConsts(ABParamGroup):
  # --------------------------
  # CLASS NAMES
  # --------------------------
  CLS_ANNOTATOR        : ABParam = newParam('Main Annotator')
  CLS_COMP_TBL         : ABParam = newParam('Component Table')
  CLS_VERT_REGION      : ABParam = newParam('Focused Image Graphics')
  CLS_MULT_REG_PLT     : ABParam = newParam('Main Image Graphics')
  CLS_COMP_MGR         : ABParam = newParam('Component Manager')
  CLS_MAIN_IMG_AREA    : ABParam = newParam('Main Image Area')
  CLS_FOCUSED_IMG_AREA : ABParam = newParam('Focused Component Image Area')
  CLS_REGION_BUF       : ABParam = newParam('Region Modification Buffer')
  # --------------------------
  # SCHEME PARAMETERS
  # --------------------------
  SCHEME_VALID_ID_COLOR     : ABParam = newParam('Validated ID Color', '0f0', 'color')
  SCHEME_NONVALID_ID_COLOR  : ABParam = newParam('Non-Validated ID Color', 'f00', 'color')
  SCHEME_BOUNDARY_COLOR     : ABParam = newParam('Component Boundary Color', 'ff0', 'color')
  SCHEME_BOUNDARY_WIDTH     : ABParam = newParam('Component Boundary Width', 2)
  SCHEME_ID_MARKER_SZ       : ABParam = newParam('ID Marker Size', 10)
  SCHEME_SELECTED_ID_BORDER : ABParam = newParam('Selected ID Border color', '00f', 'color')
  SCHEME_REG_VERT_COLOR     : ABParam = newParam('Vertex Color', '0f0', 'color')
  SCHEME_REG_FILL_COLOR     : ABParam = newParam('Fill Color', '00ff0046', 'color')

  # --------------------------
  # REGION-CREATION PARAMETERS
  # --------------------------
  PROP_MAIN_IMG_SEED_THRESH : ABParam = newParam('Seedpoint threshold in main image', 40.)
  PROP_MIN_COMP_SZ          : ABParam = newParam('Minimum New Component Size (px)', 50)
  PROP_NEW_COMP_SZ          : ABParam = newParam('New Component Side Length (px)', 30)
  PROP_EST_BOUNDS_ON_START  : ABParam = newParam('Estimate Boundaries on Image Load', False)
  PROP_MARGIN               : ABParam = newParam('Margin', 5)
  PROP_SEG_THRESH           : ABParam = newParam('Segmentation Threshold', 3.)
  PROP_FOCUSED_SEED_THRESH  : ABParam = newParam('Seedpoint Mean Threshold', 7.)
  PROP_UNDO_BUF_SZ          : ABParam = newParam('Size of Region Undo Buffer', 30)
  PROP_STEPS_BW_SAVE        : ABParam = newParam('Operations Before Buffer Save', 5)

  # --------------------------
  # SHORTCUT PARAMETERS
  # --------------------------
  SHC_CLEAR_BOUNDARIES      : ABParam = newParam('Clear Boundaries', '', 'shortcut')
  SHC_ESTIMATE_BOUNDARIES   : ABParam = newParam('Estimate Boundaries', '', 'shortcut')
  SHC_TBL_SET_SAME_AS_FIRST : ABParam = newParam('Set Table Rows Same As First', 'Ctrl+D', 'shortcut')
  SHC_TBL_SET_AS            : ABParam = newParam('Set Table Rows As...', 'Ctrl+Shift+D', 'shortcut')
  SHC_TBL_DEL_ROWS          : ABParam = newParam('Delete Table Rows', 'Del', 'shortcut')
  SHC_UNDO_MOD_REGION       : ABParam = newParam('Undo Modify Region', 'Ctrl+Z', 'shortcut')
  SHC_REDO_MOD_REGION       : ABParam = newParam('Redo Modify Region', 'Ctrl+Y', 'shortcut')

  # --------------------------
  # COMPONENT EXPORT PARAMETERS
  # --------------------------
  EXP_ONLY_VISIBLE: ABParam = newParam('Only export visible components', True)

AB_CONSTS = _ABConsts()

class AB_ENUMS(Enum):
  COMP_ADD_AS_MERGE = 'Add as Merge'
  COMP_ADD_AS_NEW   = 'Add as New'
  COMP_EXPORT_ALL: Enum = 'Export All Components'

if __name__ == '__main__':
  c = ComponentTypes().fromString('capacitor')
  print(c)