import os
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

# Preference directories
from s3a.structures import FRComplexVertices, FRParam, FRParamGroup, newParam

__all__ = ['BASE_DIR', 'MENU_OPTS_DIR', 'ICON_DIR', 'ANN_AUTH_DIR', 'QUICK_LOAD_DIR',
           'SCHEMES_DIR', 'LAYOUTS_DIR', 'TABLE_DIR', 'GEN_PROPS_DIR', 'SHORTCUTS_DIR',
           'MAIN_IMG_DIR', 'FOC_IMG_DIR',
           'DATE_FORMAT', 'REQD_TBL_FIELDS', 'COMP_CLASS_NA', 'FR_CONSTS']
BASE_DIR = Path(__file__).parent.parent
MENU_OPTS_DIR = BASE_DIR/'menuopts'
ICON_DIR = BASE_DIR/'icons'
ANN_AUTH_DIR = Path(MENU_OPTS_DIR)

DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# -----
# EDITORS
# -----
LAYOUTS_DIR = MENU_OPTS_DIR/'layouts'
QUICK_LOAD_DIR = MENU_OPTS_DIR/'quickloaders'
SCHEMES_DIR = MENU_OPTS_DIR/'schemes'
GEN_PROPS_DIR = MENU_OPTS_DIR/'regioncontrols'
SHORTCUTS_DIR = MENU_OPTS_DIR/'shortcuts'
FOC_IMG_DIR = MENU_OPTS_DIR/'focusedimg'
MAIN_IMG_DIR = MENU_OPTS_DIR/'mainimg'
TABLE_DIR = MENU_OPTS_DIR/'table'

# Ensure menuopts and layouts directories exist
LAYOUTS_DIR.mkdir(parents=True, exist_ok=True)

COMP_CLASS_NA = FRParam('Unassigned')


@dataclass
class _ReqdTableFields(FRParamGroup):
  INST_ID          : FRParam = newParam('Instance ID', -1)
  VERTICES         : FRParam = newParam('Vertices', FRComplexVertices())
  ANN_AUTHOR       : FRParam = newParam('Author', "")
  SRC_IMG_FILENAME : FRParam = newParam('Source Image Filename', "")
  ANN_TIMESTAMP    : FRParam = newParam('Timestamp', "")
  COMP_CLASS       : FRParam = newParam('Class', COMP_CLASS_NA)
REQD_TBL_FIELDS = _ReqdTableFields()


@dataclass
class _FRConsts(FRParamGroup):
  # --------------------------
  # CLASS NAMES
  # --------------------------
  CLS_ANNOTATOR        : FRParam = newParam('S3A Window')

  CLS_COMP_TBL         : FRParam = newParam('Component Table')
  CLS_COMP_MGR         : FRParam = newParam('Component Manager')
  CLS_COMP_EXPORTER    : FRParam = newParam('Component Exporter')

  CLS_VERT_IMG         : FRParam = newParam('Focused Image Graphics')
  CLS_MULT_REG_PLT     : FRParam = newParam('Main Image Graphics')

  CLS_REGION_BUF       : FRParam = newParam('Region Modification Buffer')

  CLS_IMG_AREA         : FRParam = newParam('Base Image Area')
  CLS_MAIN_IMG_AREA    : FRParam = newParam('Main Image Area')
  CLS_FOCUSED_IMG_AREA : FRParam = newParam('Focused Component Image Area')
  CLS_ROI_CLCTN        : FRParam = newParam('Draw-able Shapes')
  # --------------------------
  # SCHEME PARAMETERS
  # --------------------------
  SCHEME_USE_DARK_THEME : FRParam = newParam('Use dark theme', False)
  SCHEME_BOUND_CLR      : FRParam = newParam('Normal Boundary Color', 'ff0', 'color')
  SCHEME_SEL_BOUND_CLR  : FRParam = newParam('Selected Boundary Color', '00f', 'color')
  SCHEME_FOC_BRUSH_CLR  : FRParam = newParam('Focused Brush Color', 'f007', 'color')
  SCHEME_BOUND_WIDTH    : FRParam = newParam('Boundary Width', 7)
  SCHEME_REG_VERT_COLOR : FRParam = newParam('Vertex Color', '0f0', 'color')
  SCHEME_REG_FILL_COLOR : FRParam = newParam('Fill Color', '00ff0046', 'color')
  SCHEME_ROI_LINE_CLR   : FRParam = newParam('ROI Line Color', 'fff', 'color')
  SCHEME_ROI_LINE_WIDTH : FRParam = newParam('ROI Line Width', 1)

  # --------------------------
  # REGION-CREATION PARAMETERS
  # --------------------------
  PROP_MAIN_IMG_SEED_THRESH    : FRParam = newParam('Seedpoint Threshold in Main Image', 10.)
  PROP_MIN_COMP_SZ             : FRParam = newParam('Minimum New Component Size (px)', 50)
  PROP_NEW_COMP_SZ             : FRParam = newParam('New Component Side Length (px)', 30)
  PROP_EST_BOUNDS_ON_START     : FRParam = newParam('Estimate Boundaries on Image Load', False)
  PROP_MK_MULT_COMPS_ON_ADD    : FRParam = newParam('New Component for Each Separate Boundary', False,
                                                    helpText='When creating a new component in the '
                                                             'main image, separate boundaries'
                                                             ' will not be assigned to one single'
                                                             ' component. Rather, a new compnent'
                                                             ' will be created for each closed boundary.')
  PROP_CROP_MARGIN_PCT         : FRParam = newParam('Crop Padding (% of image size)', 15)
  PROP_FOCUSED_SEED_THRESH     : FRParam = newParam('Seedpoint Threshold in Focused Image', 7.)
  PROP_UNDO_BUF_SZ             : FRParam = newParam('Size of Undo Buffer', 300)
  PROP_STEPS_BW_SAVE           : FRParam = newParam('Operations Between Buffer Saves', 1)
  PROP_CHECK_LARGE_CHANGES     : FRParam = newParam('Save buffer during large region changes', True)
  PROP_SHOW_TBL_ON_COMP_CREATE : FRParam = newParam('Show popup table when creating component', False)

  # --------------------------
  # WINDOW TOOLS
  # --------------------------
  TOOL_MERGE_COMPS            : FRParam = newParam('Merge Selected Components',
                                                   valType='actionwithshortcut',
                                                   helpText='Merges the selected components'
                                                   ' into one, keeping all properties of'
                                                   ' the first in the selection')
  TOOL_OVERRIDE_VERTS_ACT : FRParam = newParam('Override Created Boundary',
                                              valType='actionwithshortcut',
                                              helpText='Overrides the created bounary'
                                                            ' to be the shape of the ROI, not'
                                                            ' the result of the image processor')

  # --------------------------
  # SHORTCUT PARAMETERS
  # --------------------------
  SHC_CLEAR_BOUNDARIES      : FRParam = newParam('Clear Boundaries', 'Ctrl+Alt+Shift+C', 'shortcut')
  SHC_ESTIMATE_BOUNDARIES   : FRParam = newParam('Estimate Boundaries', 'Ctrl+Alt+Shift+E', 'shortcut')
  SHC_EXPORT_COMP_LIST      : FRParam = newParam('Export Component List', 'Ctrl+S', 'shortcut')
  SHC_CLEAR_SHAPE_MAIN      : FRParam = newParam('Clear Draw Shape in Main Image', 'Esc', 'shortcut')
  SHC_CLEAR_SHAPE_FOC       : FRParam = newParam('Clear Draw Shape in Focused Image', 'Esc', 'shortcut')
  SHC_DESEL_ALL_BOUNDARIES  : FRParam = newParam('Deselect All Main Image Boundaries', 'Esc', 'shortcut')
  SHC_TBL_SET_SAME_AS_FIRST : FRParam = newParam('Set Table Rows Same As First', 'Ctrl+D', 'shortcut')
  SHC_TBL_SET_AS            : FRParam = newParam('Set Table Rows As...', 'Ctrl+Shift+D', 'shortcut')
  SHC_TBL_DEL_ROWS          : FRParam = newParam('Delete Table Rows', 'Del', 'shortcut')
  SHC_UNDO_MOD_REGION       : FRParam = newParam('Undo Modify Region', 'Ctrl+Z', 'shortcut')
  SHC_REDO_MOD_REGION       : FRParam = newParam('Redo Modify Region', 'Ctrl+Y', 'shortcut')
  SHC_ACCEPT_REGION         : FRParam = newParam('Accept and Save Focused Image Edits', '1', 'shortcut')
  SHC_DRAW_FG               : FRParam = newParam('Add to Foreground', 'Alt+D,F', 'shortcut')
  SHC_DRAW_BG               : FRParam = newParam('Add to Background', 'Alt+D,B', 'shortcut')
  SHC_DRAW_SELECT           : FRParam = newParam('Enter "Select" Mode', 'Alt+D,S', 'shortcut')
  SHC_DRAW_PAN              : FRParam = newParam('Enter "Pan" Mode', 'Alt+D,N', 'shortcut')
  SHC_DRAW_RECT             : FRParam = newParam('Enter Rect Draw Shape', 'Alt+D,R', 'shortcut')
  SHC_DRAW_POLY             : FRParam = newParam('Enter Polygon Draw Shape', 'Alt+D,Y', 'shortcut')
  SHC_DRAW_PAINT            : FRParam = newParam('Enter Paint Draw Shape', 'Alt+D,T', 'shortcut')

  # --------------------------
  # KEYBOARD MODIFIERS DURING CLICK
  # --------------------------
  MOD_MODE_SELECT  : FRParam = newParam('Enter Component Creation Mode', 'Ctrl', 'shortcut')
  MOD_MODE_EDIT    : FRParam = newParam('Enter Component Selection Mode', 'Shift', 'shortcut')

  # --------------------------
  # COMPONENT EXPORT PARAMETERS
  # --------------------------
  EXP_ONLY_VISIBLE   : FRParam = newParam('Only Export Visible Components', False)
  INCLUDE_FNAME_PATH : FRParam = newParam('Include full image path on export', False,
  None, 'Whether to just export the image name with this component group or the '
        'full absolute filepath')

  # --------------------------
  # COMPONENT EXPORT PARAMETERS
  # --------------------------
  ANN_CUR_FILE_INDICATOR: FRParam = newParam('Current image file string', 'Newly Added')

  # --------------------------
  # DRAWING
  # -------------------
  # Shapes
  DRAW_SHAPE_RECT : FRParam = newParam('Rectangle', str(ICON_DIR/'rectangle.svg'), 'icon')
  DRAW_SHAPE_POLY : FRParam = newParam('Polygon', str(ICON_DIR/'polygon.svg'), 'icon')
  DRAW_SHAPE_FREE : FRParam = newParam('Freehand')
  DRAW_SHAPE_PAINT: FRParam = newParam('Paint', str(ICON_DIR/'paint.svg'), 'icon')
  DRAW_SHAPE_NONE : FRParam = newParam('None')
  # Actions
  DRAW_ACT_ADD    : FRParam = newParam('Add to Foreground', str(ICON_DIR/'foreground.png'), 'icon')
  DRAW_ACT_REM    : FRParam = newParam('Add to Background', str(ICON_DIR/'background.png'), 'icon')
  DRAW_ACT_SELECT : FRParam = newParam('Select', str(ICON_DIR/'select.svg'), 'icon')
  DRAW_ACT_PAN    : FRParam = newParam('Pan', str(ICON_DIR/'pan.svg'), 'icon')
FR_CONSTS = _FRConsts()

# from ruamel.yaml import YAML
# yaml = YAML()
# for cls in FRParam, FRParamGroup, _FRConsts:
#   yaml.register_class(cls)
# for p in FR_CONSTS:
#   p.group = []
# p = Path('./consts.yml')
# yaml.dump(FR_CONSTS, p)