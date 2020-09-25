from dataclasses import dataclass
from enum import Enum
from pathlib import Path

__all__ = ['BASE_DIR', 'MENU_OPTS_DIR', 'ICON_DIR', 'ANN_AUTH_DIR', 'QUICK_LOAD_DIR',
           'SCHEMES_DIR', 'LAYOUTS_DIR', 'TABLE_DIR', 'GEN_PROPS_DIR', 'SHORTCUTS_DIR',
           'MAIN_IMG_DIR', 'FOC_IMG_DIR', 'APP_STATE_DIR',
           'DATE_FORMAT', 'REQD_TBL_FIELDS', 'FR_CONSTS', 'FR_ENUMS']
BASE_DIR = Path(__file__).parent
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
GEN_PROPS_DIR = MENU_OPTS_DIR/'genprops'
SHORTCUTS_DIR = MENU_OPTS_DIR/'shortcuts'
FOC_IMG_DIR = MENU_OPTS_DIR/'focusedimg'
MAIN_IMG_DIR = MENU_OPTS_DIR/'mainimg'
TABLE_DIR = MENU_OPTS_DIR/'table'
APP_STATE_DIR = Path.home()/'.s3a'


# Ensure menuopts and layouts directories exist
LAYOUTS_DIR.mkdir(parents=True, exist_ok=True)
APP_STATE_DIR.mkdir(parents=True, exist_ok=True)

class _FREnums(Enum):
  # --------------------------
  # COMPONENTS
  # --------------------------
  COMP_ADD_AS_MERGE = 'Add as Merge'
  COMP_ADD_AS_NEW   = 'Add as New'
  COMP_EXPORT_ALL   = 'Export All Components'
  COMP_RM_ALL       = "Remove All"

  # --------------------------
  # REGION CREATION
  # --------------------------
  BUFFER_UNDO       = 'Undo'
  BUFFER_REDO       = 'Redo'

  # --------------------------
  # VERTICES
  # --------------------------
  HIER_ALL_FILLED  = 'All Filled'

FR_ENUMS = _FREnums

from s3a.structures import FRComplexVertices, FRParam, FRParamGroup, newParam

@dataclass
class _ReqdTableFields(FRParamGroup):
  INST_ID          : FRParam = newParam('Instance ID', -1)
  VERTICES         : FRParam = newParam('Vertices', FRComplexVertices())
  ANN_AUTHOR       : FRParam = newParam('Author', "")
  SRC_IMG_FILENAME : FRParam = newParam('Source Image Filename', "")
  ANN_TIMESTAMP    : FRParam = newParam('Timestamp', "")
  COMP_CLASS       : FRParam = newParam('Class', 'Unassigned', 'list', limits=[])
REQD_TBL_FIELDS = _ReqdTableFields()


@dataclass
class _FRConsts(FRParamGroup):
  # --------------------------
  # CLASS NAMES
  # --------------------------
  CLS_ANNOTATOR        : FRParam = newParam('S3A Window')
  CLS_S3A_MODEL        : FRParam = newParam('S3A Model')

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
  SCHEME_USE_DARK_THEME  : FRParam = newParam('Use dark theme', False)
  SCHEME_BOUND_CLR       : FRParam = newParam('Normal Boundary Color', 'ff0', 'color')
  SCHEME_SEL_BOUND_CLR   : FRParam = newParam('Selected Boundary Color', '00f', 'color')
  SCHEME_FOC_BRUSH_CLR   : FRParam = newParam('Focused Brush Color', 'f007', 'color')
  SCHEME_BOUND_WIDTH     : FRParam = newParam('Boundary Width', 7)
  SCHEME_REG_VERT_COLOR  : FRParam = newParam('Vertex Color', '0f0', 'color')
  SCHEME_REG_FILL_COLOR  : FRParam = newParam('Fill Color', '00ff0046', 'color')
  SCHEME_ROI_LINE_CLR    : FRParam = newParam('ROI Line Color', 'fff', 'color')
  SCHEME_ROI_LINE_WIDTH  : FRParam = newParam('ROI Line Width', 1)
  SCHEME_GRID_LINE_WIDTH : FRParam = newParam('Grid Line Width', 1)
  SCHEME_GRID_CLR        : FRParam = newParam('Grid Line Color', 'fff', 'color')
  SCHEME_SHOW_GRID       : FRParam = newParam('Show Grid', False)

  # --------------------------
  # REGION-CREATION PARAMETERS
  # --------------------------
  PROP_MAIN_IMG_SEED_THRESH    : FRParam = newParam('Seedpoint Threshold in Main Image', 10.)
  PROP_MIN_COMP_SZ             : FRParam = newParam('Minimum New Component Size (px)', 50)
  PROP_NEW_COMP_SZ             : FRParam = newParam('New Component Side Length (px)', 30)
  PROP_EST_BOUNDS_ON_START     : FRParam = newParam('Estimate Boundaries on Image Load', False)
  PROP_MK_MULT_COMPS_ON_ADD    : FRParam = newParam(
    'New Component for Each Separate Boundary', False,
    helpText='When creating a new component in the main image, separate boundarieswill'
             ' not be assigned to one single component. Rather, a new compnentwill be '
             ' created for each closed boundary.')
  PROP_TREAT_MARGIN_AS_PCT     : FRParam = newParam(
    'Treat crop margin as % of image size', True,
    helpText='See note under *Crop Padding Value*')
  PROP_CROP_MARGIN_VAL         : FRParam = newParam(
    'Crop Padding Value', 15,
    helpText='If *Treat as %* is selected, then the padding will be proportionateto'
             ' the size of the focused component. Otherwise, the padding will be a'
             ' fixed pixel amount.')
  PROP_ONLY_GROW_MAIN_VB       : FRParam = newParam(
    'Disallow shrink during selection', True,
    helpText='When selecting components, this parameter determines whetherthe viewbox'
             ' shrinks only to the current selection (if *False*)or just ensures the'
             ' view contains the selected component (if *True*)')
  PROP_FOCUSED_SEED_THRESH     : FRParam = newParam('Seedpoint Threshold in Focused Image', 7.)
  PROP_UNDO_BUF_SZ             : FRParam = newParam(
    'Size of Undo Buffer', 300, helpText='Increasing the buffer size will require more memory,'
                                         'but can save a larger record of past actions.')
  PROP_STEPS_BW_SAVE           : FRParam = newParam('Operations Between Buffer Saves', 1)
  PROP_CHECK_LARGE_CHANGES     : FRParam = newParam('Save buffer during large region changes', True)
  PROP_SHOW_TBL_ON_COMP_CREATE : FRParam = newParam('Show popup table when creating component', False)
  PROP_SHOW_GUI_TOOL_BTNS      : FRParam = newParam(
    'Show tool buttons', True, helpText='Since these buttons also have right-click menu options,'
                                        ' it may be desirable to save space in the main'
                                        ' window by hiding these buttons.')
  PROP_COLS_TO_SHOW            : FRParam = newParam('Visible Table Columns', pType='group')

  # --------------------------
  # IMAGE TOOLS
  # --------------------------
  TOOL_MERGE_COMPS        : FRParam = newParam(
    'Merge Selected', 'Ctrl+Alt+G', pType='action',
    helpText='Merges the selected components into one, keeping all properties '
             'of the first in the selection')
  TOOL_SPLIT_COMPS        : FRParam = newParam(
    'Split Selected', 'Ctrl+Alt+S', pType='action',
    helpText='Makes a separate component for each distinct boundary of all selected components.')
  TOOL_OVERRIDE_VERTS_ACT : FRParam = newParam(
    'Override Created Boundary', 'Ctrl+Alt+O', pType='action',
    helpText='Overrides the most recently created bounary to be the shape of the ROI, not the result of the image processor')
  TOOL_COPY_REGIONS       : FRParam = newParam(
    'Copy Selected', 'Ctrl+Alt+C', pType='action',
    helpText='Copies the selected components. They can be pasted by <b>double-clicking</b>'
             ' on the destination location. When done copying, Click the'
             ' *Clear ROI* tool change the current draw action.')
  TOOL_MOVE_REGIONS       : FRParam = newParam(
    'Move Selected', 'Ctrl+Alt+V', pType='action',
    helpText='Moves the selected components. They can be pasted by <b>double-clicking</b>'
             ' on the destination location.')
  TOOL_CLEAR_FOC_REGION   : FRParam = newParam(
    'Clear', 'Ctrl+Shift+C', 'action',
    helpText='Clear the focused image')
  TOOL_RESET_FOC_REGION   : FRParam = newParam(
    'Reset', 'Ctrl+Shift+R', 'action',
    helpText='Reset the focused image by restoring the region mask to the last saved state')
  TOOL_FILL_FOC_REGION    : FRParam = newParam(
    'Fill', 'Ctrl+Shift+F', 'action',
    helpText='Completely fill the focused region mask')
  TOOL_ACCEPT_FOC_REGION  : FRParam = newParam(
    'Accept', 'Ctrl+Shift+A', 'action',
    helpText='Applies the focused image vertices to the corresponding component in the table')
  TOOL_CLEAR_ROI          : FRParam = newParam(
    'Clear ROI', 'Esc', 'registeredaction', helpText='Clears the current ROI', guibtn=False)
  TOOL_CLEAR_HISTORY      : FRParam = newParam(
    'Clear Processor History', 'Ctrl+Alt+C,H', 'registeredaction',
    helpText='Each time an update is made in the processor, it is saved so algorithms'
             ' can take past edits into account when performing their operations. Clearing'
             ' that history will erase algorithm knowledge of past edits.', guibtn=False)

  # --------------------------
  # WINDOW TOOLS
  # --------------------------
  TOOL_ESTIMATE_BOUNDARIES   : FRParam = newParam(
    'Estimate Boundaries', 'Ctrl+Alt+Shift+E', 'registeredaction',
    helpText='Estimates component boundaries for the whole image. This is functionally'
             ' equivalent to using a square ROI over the whole image while selecting'
             ' *New component for each separate boundary*=True', guibtn=False)
  TOOL_CLEAR_BOUNDARIES      : FRParam = newParam(
    'Clear Boundaries', 'Ctrl+Alt+Shift+C', 'registeredaction',
    helpText='Removes all components from the component table', guibtn=False)
  TOOL_EXPORT_COMP_LIST      : FRParam = newParam(
    'Export Table', 'Ctrl+S', 'registeredaction',
    helpText='Saves the component table to a file')
  TOOL_TBL_SET_SAME_AS_FIRST : FRParam = newParam(
    'Set Cells as First', 'Ctrl+D', 'registeredaction',
    helpText='Sets all cells in the selection to be the same as the first row in the selection.'
             ' See the project wiki for a detailed description')
  TOOL_TBL_SET_AS            : FRParam = newParam(
    'Set Cells As...', 'Ctrl+Shift+D', 'registeredaction',
    helpText='Sets all cells in the selection to the values specified in the popup table.'
             ' See the project wiki for a detailed description')
  TOOL_TBL_DEL_ROWS          : FRParam = newParam('Delete Table Rows', 'Del', 'registeredaction')

  # --------------------------
  # COMPONENT EXPORT PARAMETERS
  # --------------------------
  EXP_ONLY_VISIBLE   : FRParam = newParam(
    'Only Export Visible Components', False,
    helpText='If *True*, only components showing on the main image will be included in'
             ' file exports.')
  INCLUDE_FNAME_PATH : FRParam = newParam('Include full image path on export', False,
  None, 'If *False*, only the image name is included in export information, e.g. <img>.png.'
        ' Otherwise, the full filepath is included, e.g. "C:/Users/.../<img>.png')

  # --------------------------
  # COMPONENT EXPORT PARAMETERS
  # --------------------------
  ANN_CUR_FILE_INDICATOR: FRParam = newParam('Current image file string', 'Newly Added')

  # --------------------------
  # DRAWING
  # -------------------
  # Shapes
  DRAW_SHAPE_RECT : FRParam = newParam(
    'Activate "Rectangular" draw shape', 'Ctrl+Alt+D,R', 'registeredaction',
    icon=str(ICON_DIR/'rectangle.svg'))
  DRAW_SHAPE_POLY : FRParam = newParam(
    'Activate "Polygon" draw shape', 'Ctrl+Alt+D,Y', 'registeredaction',
    icon=str(ICON_DIR/'polygon.svg'))
  DRAW_SHAPE_FREE : FRParam = newParam('Activate "Freehand" draw shape')
  DRAW_SHAPE_PAINT: FRParam = newParam(
    'Activate "Paint" draw shape', 'Ctrl+Alt+D,T', icon=str(ICON_DIR/'paint.svg'))
  DRAW_SHAPE_NONE : FRParam = newParam('None')
  # Actions
  DRAW_ACT_ADD    : FRParam = newParam(
    'Activate "Add to Foreground" action', 'Ctrl+Alt+D,F', 'registeredaction',
    icon=str(ICON_DIR/'foreground.png'),
    helpText='When an ROI is created, the image processor will attempt to make a new'
             ' component at that location. Right-click and drag to pan.')
  DRAW_ACT_REM    : FRParam = newParam(
    'Activate "Add to Background" action', 'Ctrl+Alt+D, B', 'registeredaction',
    icon=str(ICON_DIR/'background.png'),
    helpText='When an ROI is created, the image processor will attempt to take the enclosed'
             ' area away from the current component shape. Right-click and drag to pan.')
  DRAW_ACT_SELECT : FRParam = newParam(
    'Activate "Select" draw action', 'Ctrl+Alt+D, S', 'registeredaction', icon=str(ICON_DIR/'select.svg'),
    helpText='When component boundaries are enclosed by this ROI, they will be selected'
             ' in the component table. Right-click and drag to pan.')
  DRAW_ACT_PAN    : FRParam = newParam(
    'Activate "Pan" draw action', 'Ctrl+Alt+D,P', 'registeredaction', icon=str(ICON_DIR/'pan.svg'),
    helpText='No ROI will be drawn in this mode. Right- or left-click and drag to pan.')
FR_CONSTS = _FRConsts()

# from ruamel.yaml import YAML
# yaml = YAML()
# for cls in FRParam, FRParamGroup, _FRConsts:
#   yaml.register_class(cls)
# for p in FR_CONSTS:
#   p.group = []
# p = Path('./consts.yml')
# yaml.dump(FR_CONSTS, p)