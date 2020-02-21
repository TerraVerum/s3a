import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

# Preference directories
from Annotator.params import FRParam, FRParamGroup, newParam

BASE_DIR = os.path.dirname(Path(__file__).absolute())
ANN_AUTH_DIR = os.path.join(BASE_DIR, 'MenuOpts', '')
SCHEMES_DIR = os.path.join(BASE_DIR, 'MenuOpts', 'Schemes', '')
LAYOUTS_DIR = os.path.join(BASE_DIR, 'MenuOpts', 'Layouts', '')
FILTERS_DIR = os.path.join(BASE_DIR, 'MenuOpts', 'Filters', '')
GEN_PROPS_DIR = os.path.join(BASE_DIR, 'MenuOpts', 'RegionControls', '')
SHORTCUTS_DIR = os.path.join(BASE_DIR, 'MenuOpts', 'Shortcuts', '')
CLICK_MODIFIERS_DIR = os.path.join(BASE_DIR, 'MenuOpts', 'Modifiers', '')

# Ensure these directories exist
for curDir in [SCHEMES_DIR, LAYOUTS_DIR, FILTERS_DIR,
               GEN_PROPS_DIR, SHORTCUTS_DIR, CLICK_MODIFIERS_DIR]:
  Path(curDir).mkdir(parents=True, exist_ok=True)


@dataclass
class ComponentTypes(FRParamGroup):
  CAP   : FRParam = newParam('Capacitor')
  RES   : FRParam = newParam('Resistor')
  IND   : FRParam = newParam('Inductor')
  TRANS : FRParam = newParam('Transistor')
  IC    : FRParam = newParam('IC')
  OTHER : FRParam = newParam('Other')
  N_A   : FRParam = newParam('Unassigned')

TEMPLATE_COMP_TYPES = ComponentTypes()


@dataclass
class CompParams(FRParamGroup):
  # These 3 params MUST exist in the component
  INST_ID         : FRParam = newParam('Instance ID', -1)
  VERTICES        : FRParam = newParam('Vertices', np.ones((1, 2)) * np.nan)
  VALIDATED       : FRParam = newParam('Validated', False)
  ANN_AUTHOR      : FRParam = newParam('Author', "")
  ANN_FILENAME    : FRParam = newParam('Filename', "")
  ANN_TIMESTAMP   : FRParam = newParam('Timestamp', "")
  DEV_TEXT        : FRParam = newParam('Device Text', '')
  DEV_TYPE        : FRParam = newParam('Device Type', TEMPLATE_COMP_TYPES.N_A)
  BOARD_TEXT      : FRParam = newParam('Board Text', '')
  LOGO            : FRParam = newParam('Logo', '')
  NOTES           : FRParam = newParam('Notes', '')

TEMPLATE_COMP = CompParams()


@dataclass
class _FRConsts(FRParamGroup):
  # --------------------------
  # CLASS NAMES
  # --------------------------
  CLS_ANNOTATOR        : FRParam = newParam('Main Annotator')

  CLS_COMP_TBL         : FRParam = newParam('Component Table')
  CLS_COMP_MGR         : FRParam = newParam('Component Manager')

  CLS_VERT_REGION      : FRParam = newParam('Focused Image Graphics')
  CLS_MULT_REG_PLT     : FRParam = newParam('Main Image Graphics')

  CLS_REGION_BUF       : FRParam = newParam('Region Modification Buffer')

  CLS_IMG_AREA         : FRParam = newParam('Base Image Area')
  CLS_MAIN_IMG_AREA    : FRParam = newParam('Main Image Area')
  CLS_FOCUSED_IMG_AREA : FRParam = newParam('Focused Component Image Area')
  # --------------------------
  # SCHEME PARAMETERS
  # --------------------------
  SCHEME_VALID_ID_COLOR     : FRParam = newParam('Validated ID Color', '0f0', 'color')
  SCHEME_NONVALID_ID_COLOR  : FRParam = newParam('Non-Validated ID Color', 'f00', 'color')
  SCHEME_BOUNDARY_COLOR     : FRParam = newParam('Component Boundary Color', 'ff0', 'color')
  SCHEME_BOUNDARY_WIDTH     : FRParam = newParam('Component Boundary Width', 2)
  SCHEME_ID_MARKER_SZ       : FRParam = newParam('ID Marker Size', 10)
  SCHEME_SELECTED_ID_BORDER : FRParam = newParam('Selected ID Border color', '00f', 'color')
  SCHEME_REG_VERT_COLOR     : FRParam = newParam('Vertex Color', '0f0', 'color')
  SCHEME_REG_FILL_COLOR     : FRParam = newParam('Fill Color', '00ff0046', 'color')

  # --------------------------
  # REGION-CREATION PARAMETERS
  # --------------------------
  PROP_MAIN_IMG_SEED_THRESH : FRParam = newParam('Seedpoint Threshold in Main Image', 10.)
  PROP_MIN_COMP_SZ          : FRParam = newParam('Minimum New Component Size (px)', 50)
  PROP_NEW_COMP_SZ          : FRParam = newParam('New Component Side Length (px)', 30)
  PROP_EST_BOUNDS_ON_START  : FRParam = newParam('Estimate Boundaries on Image Load', False)
  PROP_MARGIN               : FRParam = newParam('Margin', 5)
  PROP_SEG_THRESH           : FRParam = newParam('Segmentation Threshold', 3.)
  PROP_FOCUSED_SEED_THRESH  : FRParam = newParam('Seedpoint Threshold in Focused Image', 7.)
  PROP_UNDO_BUF_SZ          : FRParam = newParam('Size of Region Undo Buffer', 30)
  PROP_STEPS_BW_SAVE        : FRParam = newParam('Operations Between Buffer Saves', 5)

  # --------------------------
  # SHORTCUT PARAMETERS
  # --------------------------
  SHC_CLEAR_BOUNDARIES      : FRParam = newParam('Clear Boundaries', '', 'shortcut')
  SHC_ESTIMATE_BOUNDARIES   : FRParam = newParam('Estimate Boundaries', '', 'shortcut')
  SHC_DESEL_ALL_BOUNDARIES  : FRParam = newParam('Deselect All Main Image Boundaries', 'Esc', 'shortcut')
  SHC_TBL_SET_SAME_AS_FIRST : FRParam = newParam('Set Table Rows Same As First', 'Ctrl+D', 'shortcut')
  SHC_TBL_SET_AS            : FRParam = newParam('Set Table Rows As...', 'Ctrl+Shift+D', 'shortcut')
  SHC_TBL_DEL_ROWS          : FRParam = newParam('Delete Table Rows', 'Del', 'shortcut')
  SHC_UNDO_MOD_REGION       : FRParam = newParam('Undo Modify Region', 'Ctrl+Z', 'shortcut')
  SHC_REDO_MOD_REGION       : FRParam = newParam('Redo Modify Region', 'Ctrl+Y', 'shortcut')
  SHC_ACCEPT_REGION         : FRParam = newParam('Accept and Save Region Edits', '1', 'shortcut')
  SHC_TOGGLE_REG_MODE       : FRParam = newParam('Toggle Between Add/Remove Region', 'Ctrl+T', 'shortcut')

  # --------------------------
  # KEYBOARD MODIFIERS DURING CLICK
  # --------------------------
  MOD_MODE_SELECT  : FRParam = newParam('Enter Component Creation Mode', 'Ctrl', 'shortcut')
  MOD_MODE_EDIT    : FRParam = newParam('Enter Component Selection Mode', 'Shift', 'shortcut')

  # --------------------------
  # COMPONENT EXPORT PARAMETERS
  # --------------------------
  EXP_ONLY_VISIBLE: FRParam = newParam('Only Export Visible Components', True)

  # --------------------------
  # COMPONENT EXPORT PARAMETERS
  # --------------------------
  ANN_CUR_FILE_INDICATOR: FRParam = newParam('New')

  # --------------------------
  # DRAWING
  # -------------------
  # Shapes
  DRAW_SHAPE_RECT: FRParam = newParam('Rect')
  DRAW_SHAPE_POLY: FRParam = newParam('Polygon')
  DRAW_SHAPE_FREE: FRParam = newParam('Freehand')
  DRAW_SHAPE_PAINT: FRParam = newParam('Paint')
  DRAW_SHAPE_NONE: FRParam = newParam('None')
  # Actions
  DRAW_ACT_ADD: FRParam = newParam('Add')
  DRAW_ACT_REM: FRParam = newParam('Remove')
  DRAW_ACT_SELECT: FRParam = newParam('Select')
  DRAW_ACT_PAN: FRParam = newParam('Pan')

FR_CONSTS = _FRConsts()

class _FREnums(Enum):
  # --------------------------
  # COMPONENTS
  # --------------------------
  COMP_ADD_AS_MERGE = 'Add as Merge'
  COMP_ADD_AS_NEW   = 'Add as New'
  COMP_EXPORT_ALL   = 'Export All Components'

  # --------------------------
  # REGION CREATION
  # --------------------------
  BUFFER_UNDO       = 'Undo'
  BUFFER_REDO       = 'Redo'
FR_ENUMS = _FREnums

if __name__ == '__main__':
  c = ComponentTypes().fromString('capacitor')
  print(c)