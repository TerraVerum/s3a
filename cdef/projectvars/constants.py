import os
from dataclasses import dataclass
from pathlib import Path

# Preference directories
from ..structures import FRComplexVertices, FRParam, FRParamGroup, newParam

__all__ = ['BASE_DIR', 'MENU_OPTS_DIR', 'ICON_DIR', 'ANN_AUTH_DIR',
           'SCHEMES_DIR', 'LAYOUTS_DIR', 'FILTERS_DIR', 'GEN_PROPS_DIR', 'SHORTCUTS_DIR',
           'FR_CONSTS', 'TEMPLATE_COMP_TYPES', 'TEMPLATE_COMP', 'CompParams', 'ComponentTypes']
BASE_DIR = Path(__file__).parent.parent.absolute()
MENU_OPTS_DIR = os.path.join(BASE_DIR, 'menuopts', '')
ICON_DIR = os.path.join(BASE_DIR, 'icons', '')
ANN_AUTH_DIR = os.path.join(MENU_OPTS_DIR)

# -----
# EDITORS
# -----
SCHEMES_DIR = os.path.join(MENU_OPTS_DIR, 'schemes', '')
LAYOUTS_DIR = os.path.join(MENU_OPTS_DIR, 'layouts', '')
FILTERS_DIR = os.path.join(MENU_OPTS_DIR, 'filters', '')
GEN_PROPS_DIR = os.path.join(MENU_OPTS_DIR, 'regioncontrols', '')
SHORTCUTS_DIR = os.path.join(MENU_OPTS_DIR, 'shortcuts', '')

# Ensure menuopts and layouts directories exist
Path(LAYOUTS_DIR).mkdir(exist_ok=True)


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
  VERTICES        : FRParam = newParam('Vertices', FRComplexVertices())
  VALIDATED       : FRParam = newParam('Validated', False)
  ANN_AUTHOR      : FRParam = newParam('Author', "")
  ANN_FILENAME    : FRParam = newParam('Source Image Filename', "")
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
  CLS_ANNOTATOR        : FRParam = newParam('Main MainWindow')

  CLS_COMP_TBL         : FRParam = newParam('Component Table')
  CLS_COMP_MGR         : FRParam = newParam('Component Manager')

  CLS_VERT_IMG         : FRParam = newParam('Focused Image Graphics')
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
  PROP_SEG_THRESH           : FRParam = newParam('Segmentation Threshold', 0.)
  PROP_FOCUSED_SEED_THRESH  : FRParam = newParam('Seedpoint Threshold in Focused Image', 7.)
  PROP_UNDO_BUF_SZ          : FRParam = newParam('Size of Region Undo Buffer', 30)
  PROP_STEPS_BW_SAVE        : FRParam = newParam('Operations Between Buffer Saves', 1)
  PROP_CHECK_LARGE_CHANGES  : FRParam = newParam('Save buffer during large region changes', True)

  # --------------------------
  # SHORTCUT PARAMETERS
  # --------------------------
  SHC_CLEAR_BOUNDARIES      : FRParam = newParam('Clear Boundaries', 'Ctrl+C', 'shortcut')
  SHC_ESTIMATE_BOUNDARIES   : FRParam = newParam('Estimate Boundaries', 'Ctrl+E', 'shortcut')
  SHC_CLEAR_SHAPE_MAIN      : FRParam = newParam('Clear Draw Shape in Main Image', 'Esc', 'shortcut')
  SHC_CLEAR_SHAPE_FOC       : FRParam = newParam('Clear Draw Shape in Focused Image', 'Esc', 'shortcut')
  SHC_DESEL_ALL_BOUNDARIES  : FRParam = newParam('Deselect All Main Image Boundaries', 'Esc', 'shortcut')
  SHC_TBL_SET_SAME_AS_FIRST : FRParam = newParam('Set Table Rows Same As First', 'Ctrl+D', 'shortcut')
  SHC_TBL_SET_AS            : FRParam = newParam('Set Table Rows As...', 'Ctrl+Shift+D', 'shortcut')
  SHC_TBL_DEL_ROWS          : FRParam = newParam('Delete Table Rows', 'Del', 'shortcut')
  SHC_UNDO_MOD_REGION       : FRParam = newParam('Undo Modify Region', 'Ctrl+Z', 'shortcut')
  SHC_REDO_MOD_REGION       : FRParam = newParam('Redo Modify Region', 'Ctrl+Y', 'shortcut')
  SHC_ACCEPT_REGION         : FRParam = newParam('Accept and Save Region Edits', '1', 'shortcut')
  SHC_DRAW_FG               : FRParam = newParam('Add to Foreground', 'Ctrl+D,F', 'shortcut')
  SHC_DRAW_BG               : FRParam = newParam('Add to Background', 'Ctrl+D,B', 'shortcut')
  SHC_DRAW_SELECT           : FRParam = newParam('Enter "Select" Mode', 'Ctrl+D,S', 'shortcut')
  SHC_DRAW_PAN              : FRParam = newParam('Enter "Pan" Mode', 'Ctrl+D,N', 'shortcut')
  SHC_DRAW_RECT             : FRParam = newParam('Enter Rect Draw Shape', 'Ctrl+D,R', 'shortcut')
  SHC_DRAW_POLY             : FRParam = newParam('Enter Polygon Draw Shape', 'Ctrl+D,Y', 'shortcut')
  SHC_DRAW_PAINT            : FRParam = newParam('Enter Paint Draw Shape', 'Ctrl+D,T', 'shortcut')

  # --------------------------
  # KEYBOARD MODIFIERS DURING CLICK
  # --------------------------
  MOD_MODE_SELECT  : FRParam = newParam('Enter Component Creation Mode', 'Ctrl', 'shortcut')
  MOD_MODE_EDIT    : FRParam = newParam('Enter Component Selection Mode', 'Shift', 'shortcut')

  # --------------------------
  # COMPONENT EXPORT PARAMETERS
  # --------------------------
  EXP_ONLY_VISIBLE   : FRParam = newParam('Only Export Visible Components', True)
  INCLUDE_FNAME_PATH : FRParam = newParam('Include full image path on export', False,
  None, 'Whether to just export the image name with this component group or the '
        'full absolute filepath')

  # --------------------------
  # COMPONENT EXPORT PARAMETERS
  # --------------------------
  ANN_CUR_FILE_INDICATOR: FRParam = newParam('Current image file string', 'New')

  # --------------------------
  # DRAWING
  # -------------------
  # Shapes
  DRAW_SHAPE_RECT : FRParam = newParam('Rectangle', f'{ICON_DIR}rectangle.svg', 'icon')
  DRAW_SHAPE_POLY : FRParam = newParam('Polygon', f'{ICON_DIR}polygon.svg', 'icon')
  DRAW_SHAPE_FREE : FRParam = newParam('Freehand')
  DRAW_SHAPE_PAINT: FRParam = newParam('Paint', f'{ICON_DIR}paint.svg', 'icon')
  DRAW_SHAPE_NONE : FRParam = newParam('None')
  # Actions
  DRAW_ACT_ADD    : FRParam = newParam('Add to Foreground', f'{ICON_DIR}foreground.png', 'icon')
  DRAW_ACT_REM    : FRParam = newParam('Add to Background', f'{ICON_DIR}background.png', 'icon')
  DRAW_ACT_SELECT : FRParam = newParam('Select', f'{ICON_DIR}select.svg', 'icon')
  DRAW_ACT_PAN    : FRParam = newParam('Pan', f'{ICON_DIR}pan.svg', 'icon')
FR_CONSTS = _FRConsts()