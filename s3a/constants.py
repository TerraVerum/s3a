from dataclasses import dataclass
from pathlib import Path

__all__ = ['BASE_DIR', 'MENU_OPTS_DIR', 'ICON_DIR', 'QUICK_LOAD_DIR',
           'SCHEMES_DIR', 'LAYOUTS_DIR', 'TABLE_DIR', 'GEN_PROPS_DIR', 'SHORTCUTS_DIR',
           'SHORTCUT_BASE', 'MAIN_IMG_DIR', 'APP_STATE_DIR', 'CFG_DIR', 'IMG_PROC_DIR', 'MULT_PRED_DIR',
           'REQD_TBL_FIELDS', 'PRJ_CONSTS', 'PRJ_ENUMS', 'PROJ_FILE_TYPE',
           'IO_TEMPLATES_DIR', 'PROJ_BASE_TEMPLATE']

import utilitys.constants
from utilitys import fns

CODE_DIR = Path(__file__).parent
BASE_DIR = Path.home()/'.s3a'

CFG_DIR = CODE_DIR/'cfg'
ICON_DIR = CODE_DIR/'icons'
MENU_OPTS_DIR = BASE_DIR/'menuopts'
IO_TEMPLATES_DIR = CODE_DIR/'compio/templates'

# -----
# EDITORS
# -----
LAYOUTS_DIR = MENU_OPTS_DIR/'layouts'
QUICK_LOAD_DIR = MENU_OPTS_DIR/'quickloader'
SCHEMES_DIR = MENU_OPTS_DIR/'schemes'
GEN_PROPS_DIR = MENU_OPTS_DIR/'genprops'
SHORTCUTS_DIR = MENU_OPTS_DIR/'shortcuts'
MAIN_IMG_DIR = MENU_OPTS_DIR/'mainimg'
IMG_PROC_DIR = MENU_OPTS_DIR/'imageproc'
MULT_PRED_DIR = MENU_OPTS_DIR/'multiproc'
TABLE_DIR = MENU_OPTS_DIR/'table'
APP_STATE_DIR = BASE_DIR


PROJ_BASE_TEMPLATE = CFG_DIR/'projectcfg.yml'
PROJ_FILE_TYPE = 's3aprj'

# Ensure menuopts and layouts directories exist
LAYOUTS_DIR.mkdir(parents=True, exist_ok=True)
APP_STATE_DIR.mkdir(parents=True, exist_ok=True)

SHORTCUT_BASE = 'Ctrl+Q'

class PRJ_ENUMS:
  # --------------------------
  # COMPONENTS
  # --------------------------
  COMP_ADD_AS_MERGE = 'merge'
  COMP_ADD_AS_NEW   = 'new'
  COMP_RM_ALL       = None

  # --------------------------
  # REGION CREATION
  # --------------------------
  BUFFER_UNDO       = 'Undo'
  BUFFER_REDO       = 'Redo'

  # --------------------------
  # VERTICES
  # --------------------------
  HIER_ALL_FILLED  = 'All Filled'

  # --------------------------
  # COMMON DATAFRAME FIELDS
  # --------------------------
  FIELD_SELECTED   = 'selected'
  FIELD_FOCUSED    = 'focused'
  FIELD_LABEL      = 'label'

  # --------------------------
  # I/O SPECIFICATIONS
  # --------------------------
  IO_EXPORT = 'export'
  IO_IMPORT = 'import'
  IO_ROUND_TRIP = 'round trip'
  IO_META_MAPPING = 'mapping'
  IO_META_FIELD = 'field'

  # --------------------------
  # LOGGING / FEEDBACK CAPABILITIES
  # --------------------------
  LOG_GUI = 'gui'
  LOG_TERM = 'term'
  LOG_NONE = 'none'
  LOG_FILE = 'file'
  LOG_LVL_ATTN = utilitys.constants.PrjEnums.LOG_LVL_ATTN

  # --------------------------
  # PROCESSING
  # --------------------------
  PROC_ADD_TOP  = 'top'
  PROC_ADD_PRIM = 'primitive'
  PROC_NO_ADD   = None

  HISTORY_UNSPECIFIED = 0
  HISTORY_BACKGROUND = 1
  HISTORY_FOREGROUND = 2
  
from s3a.structures import ComplexXYVertices, PrjParam, PrjParamGroup, newParam

class _ReqdTableFields:
  _extraRequired = []

  def __init__(self):
    self._iterFields = []

    # Can't use IOTemplateManager without creating cyclic reference
    fields = fns.attemptFileLoad(IO_TEMPLATES_DIR/'s3a.tblcfg')['fields']

    ctorItems = [{'name': kk, **vv} for kk, vv in fields.items()]
    ii = 0
    def constructNext():
      nonlocal ii
      ret = PrjParam(**ctorItems[ii])
      ii += 1
      self._iterFields.append(ret)
      return ret

    self.INST_ID = constructNext()
    self.SRC_IMG_FILENAME = constructNext()
    self.VERTICES = constructNext()
    # Special case: Vertices is non-primitive type
    self.VERTICES.value = ComplexXYVertices()

  def addField(self, field: PrjParam):
    if field not in self:
      self._extraRequired.append(field)

  def removeField(self, field: PrjParam):
    if field in self._extraRequired:
      self._extraRequired.remove(field)

  def __iter__(self):
    for lst in self._iterFields, self._extraRequired:
      yield from lst.__iter__()


REQD_TBL_FIELDS = _ReqdTableFields()


@dataclass
class _PrjConsts(PrjParamGroup):
  # --------------------------
  # CLASS NAMES
  # --------------------------
  CLS_ANNOTATOR        : PrjParam = newParam('S3A Window')
  CLS_S3A_MODEL        : PrjParam = newParam('S3A Model')

  CLS_COMP_TBL         : PrjParam = newParam('Component Table')
  CLS_COMP_MGR         : PrjParam = newParam('Component Manager')
  CLS_COMP_EXPORTER    : PrjParam = newParam('Component Exporter')

  CLS_VERT_IMG         : PrjParam = newParam('Focused Image Graphics')
  CLS_MULT_REG_PLT     : PrjParam = newParam('Main Image')
  CLS_ROI_CLCTN        : PrjParam = newParam('ROI Shapes')

  CLS_REGION_BUF       : PrjParam = newParam('Region Modification Buffer')

  CLS_IMG_AREA         : PrjParam = newParam('Base Image Area')
  CLS_MAIN_IMG_AREA    : PrjParam = newParam('Main Image Area')
  CLS_FOCUSED_IMG_AREA : PrjParam = newParam('Focused Image Area')
  # --------------------------
  # SCHEME PARAMETERS
  # --------------------------
  SCHEME_USE_DARK_THEME  : PrjParam = newParam('Use dark theme', False)
  SCHEME_BOUND_CLR       : PrjParam = newParam('Normal Boundary Color', '#ff0', 'color')
  SCHEME_SEL_BOUND_CLR   : PrjParam = newParam('Selected Boundary Color', '#00f', 'color')
  SCHEME_FOC_BRUSH_CLR   : PrjParam = newParam('Focused Brush Color', '#f007', 'color')
  SCHEME_BOUND_WIDTH     : PrjParam = newParam('Boundary Width', 7)
  SCHEME_REG_VERT_COLOR  : PrjParam = newParam('Vertex Color', '#0f0', 'color')
  SCHEME_REG_FILL_COLOR  : PrjParam = newParam('Fill Color', '#00ff0046', 'color')
  SCHEME_ROI_LINE_CLR    : PrjParam = newParam('ROI Line Color', '#fff', 'color')
  SCHEME_ROI_LINE_WIDTH  : PrjParam = newParam('ROI Line Width', 1)
  SCHEME_GRID_LINE_WIDTH : PrjParam = newParam('Grid Line Width', 1)
  SCHEME_GRID_CLR        : PrjParam = newParam('Grid Line Color', '#fff', 'color')
  SCHEME_SHOW_GRID       : PrjParam = newParam('Show Grid', False)
  SCHEME_LBL_COL         : PrjParam = newParam('Labeling Column', REQD_TBL_FIELDS.INST_ID.name, pType='list', limits=[])

  # --------------------------
  # REGION-CREATION PARAMETERS
  # --------------------------
  PROP_MIN_COMP_SZ             : PrjParam = newParam('Minimum New Component Size (px)', 50)
  PROP_UNDO_BUF_SZ             : PrjParam = newParam('Undo Buffer Size', 300)
  PROP_COMP_SEL_BHV            : PrjParam = newParam(
    'Component Selection Behavior', 'Entire Component', 'list', limits=['Boundary Only', 'Entire Component'],
    helpText='When `Boundary Only`, components are selected when clicking on their boundaries.\n'
             'When `Entire Component`, clicking anywhere within the component'
             ' will select it')
  PROP_FIELD_INFO_ON_SEL      : PrjParam = newParam(
    'Show field info after selection', False,
    helpText='Whether to show field information in the main image'
             ' every time the current selection changes')
  PROP_VERT_SORT_BHV           : PrjParam = newParam(
    'Vertices Sorting', 'X First', 'list', limits=['X First', 'Y First'],
    helpText='Whether to sort first by X or Y when sorting a vertices-like field')
  PROP_SHOW_TBL_ON_COMP_CREATE : PrjParam = newParam('Show popup table when creating component', False)
  PROP_SCALE_PEN_WIDTH : PrjParam = newParam('Scale pen width to image pixel size', True)

  PROP_COLLECT_USR_METRICS: PrjParam = newParam('Collect user metrics', False, helpText='Collect user metrics for smarter segmentation algorithms')
  # --------------------------
  # MISC TOOLS
  # --------------------------
  TOOL_UNDO               : PrjParam =  newParam('Undo', 'Ctrl+Z')
  TOOL_REDO               : PrjParam =  newParam('Redo', 'Ctrl+Y')
  # --------------------------
  # IMAGE TOOLS
  # --------------------------
  TOOL_MERGE_COMPS        : PrjParam = newParam('Merge Selected', f'{SHORTCUT_BASE},S,M', icon=str(ICON_DIR/'merge.svg'))
  TOOL_SPLIT_COMPS        : PrjParam = newParam('Split Selected', f'{SHORTCUT_BASE},S,S', icon=str(ICON_DIR/'split.svg'))
  TOOL_REM_OVERLAP        : PrjParam = newParam('Remove Component Overlap', f'{SHORTCUT_BASE},S,E', icon=str(ICON_DIR/'mutex.svg'))
  TOOL_COPY_REGIONS       : PrjParam = newParam('Copy Selected', f'{SHORTCUT_BASE},S,C', icon=str(ICON_DIR/'copy.svg'))
  TOOL_MOVE_REGIONS       : PrjParam = newParam('Move Selected', f'{SHORTCUT_BASE},S,V', icon=str(ICON_DIR/'move.svg'))
  TOOL_CLEAR_FOC_REGION   : PrjParam = newParam('Clear', f'{SHORTCUT_BASE},V,C', icon=str(ICON_DIR/'clear.svg'))
  TOOL_RESET_FOC_REGION   : PrjParam = newParam('Reset', f'{SHORTCUT_BASE},V,R', icon=str(ICON_DIR/'reset.svg'))
  TOOL_FILL_FOC_REGION    : PrjParam = newParam('Fill', f'{SHORTCUT_BASE},V,F', icon=str(ICON_DIR/'fill.svg'))
  TOOL_ACCEPT_FOC_REGION  : PrjParam = newParam('Accept', 'Ctrl+Shift+A', icon=str(ICON_DIR/'accept.svg'))
  TOOL_CLEAR_ROI          : PrjParam = newParam('Clear ROI', 'Esc')
  TOOL_CLEAR_HISTORY      : PrjParam = newParam('Clear Processor History', f'{SHORTCUT_BASE},V,H', icon=str(ICON_DIR/'clear_history.svg'))
  TOOL_PROC_ANALYTICS     : PrjParam = newParam('Show Analytics', icon=str(ICON_DIR/'analytics.svg'))
  TOOL_RESET_ZOOM         : PrjParam = newParam('Reset Zoom', f'{SHORTCUT_BASE},Z,R', icon=str(ICON_DIR/'reset_zoom.svg'))

  # --------------------------
  # WINDOW TOOLS
  # --------------------------
  TOOL_ESTIMATE_BOUNDARIES   : PrjParam = newParam('Estimate Boundaries', 'Ctrl+Alt+Shift+E')
  TOOL_CLEAR_BOUNDARIES      : PrjParam = newParam('Clear Boundaries', 'Ctrl+Alt+Shift+C')
  TOOL_EXPORT_COMP_LIST      : PrjParam = newParam('Export Current Table', f'{SHORTCUT_BASE},E,T')
  TOOL_TBL_SET_SAME_AS_FIRST : PrjParam = newParam('Set Cells as First', 'Ctrl+D')
  TOOL_TBL_SET_AS            : PrjParam = newParam('Set Cells As...', 'Ctrl+Shift+D')
  TOOL_TBL_DEL_ROWS          : PrjParam = newParam('Delete Table Rows', 'Del')
  TOOL_TBL_ZOOM_TO_COMPS     : PrjParam = newParam('Zoom to Selection  ', f'{SHORTCUT_BASE},Z,S')

  # --------------------------
  # PROJECT
  # --------------------------
  TOOL_PROJ_SAVE     : PrjParam = newParam('Save', 'Ctrl+S')
  TOOL_PROJ_OPEN     : PrjParam = newParam('Open Project', f'{SHORTCUT_BASE},P,O')
  TOOL_PROJ_OPEN_IMG : PrjParam = newParam('Open Project Image', f'{SHORTCUT_BASE},I,O')
  TOOL_PROJ_CREATE   : PrjParam = newParam('Create Project', f'{SHORTCUT_BASE},P,C')
  TOOL_PROJ_ADD_IMG  : PrjParam = newParam('Add New Image', f'{SHORTCUT_BASE},I,A')
  TOOL_PROJ_ADD_ANN  : PrjParam = newParam('Add New Annotation', f'{SHORTCUT_BASE},A,A')
  TOOL_PROJ_SETTINGS : PrjParam = newParam('Project Settings...', f'{SHORTCUT_BASE},P,S')
  TOOL_PROJ_EXPORT   : PrjParam = newParam('Export...', f'{SHORTCUT_BASE},P,E')
  TOOL_AUTOSAVE      : PrjParam = newParam('Autosave...', f'{SHORTCUT_BASE},A,O')

  # --------------------------
  # GLOBAL PREDICTIONS
  # --------------------------
  TOOL_PRED_SEL     : PrjParam = newParam('Predict From Selection', f'{SHORTCUT_BASE},M,P', icon=str(ICON_DIR/'predict.svg'))


  # --------------------------
  # COMPONENT EXPORT PARAMETERS
  # --------------------------
  EXP_ONLY_VISIBLE   : PrjParam = newParam(
    'Only Export Visible Components', False,
    helpText='If *True*, only components showing on the main image will be included in'
             ' file exports.')
  INCLUDE_FNAME_PATH : PrjParam = newParam('Include full image path on export', False,
  None, 'If *False*, only the image name is included in export information, e.g. <img>.png.'
        ' Otherwise, the full filepath is included, e.g. "C:/Users/.../<img>.png')

  # --------------------------
  # DRAWING
  # -------------------
  # Modes
  DRAW_MODE_FOCUSED : PrjParam = newParam(
    'Activate "Edit" draw mode', f'{SHORTCUT_BASE},D,E', 'registeredaction',
    icon=str(ICON_DIR/'edit.svg'))

  # Shapes
  DRAW_SHAPE_RECT : PrjParam = newParam(
    'Activate "Rectangular" draw shape', f'{SHORTCUT_BASE},D,R', 'registeredaction',
    icon=str(ICON_DIR/'rectangle.svg'))
  DRAW_SHAPE_POLY : PrjParam = newParam(
    'Activate "Polygon" draw shape', f'{SHORTCUT_BASE},D,Y', 'registeredaction',
    icon=str(ICON_DIR/'polygon.svg'))
  DRAW_SHAPE_ELLIPSE : PrjParam = newParam(
    'Activate "Ellipse" draw shape', f'{SHORTCUT_BASE},D,L', 'registeredaction',
    icon=str(ICON_DIR/'ellipse.svg'))
  DRAW_SHAPE_FREE : PrjParam = newParam('Activate "Freehand" draw shape', f'{SHORTCUT_BASE},D,H',
                                        icon=str(ICON_DIR/'freehand.svg'))
  DRAW_SHAPE_POINT: PrjParam = newParam(
    'Activate "Point" draw shape', f'{SHORTCUT_BASE},D,N', icon=str(ICON_DIR / 'point.svg'))
  DRAW_SHAPE_NONE : PrjParam = newParam('None')

  # Actions
  DRAW_ACT_CREATE    : PrjParam = newParam(
    'Activate "Create Component" action', f'{SHORTCUT_BASE},D,C', 'registeredaction',
    icon=str(ICON_DIR/'create.svg'),
    helpText='When an ROI is created, the image processor will attempt to make a new'
             ' component at that location. Right-click and drag to pan.')
  DRAW_ACT_ADD    : PrjParam = newParam(
    'Activate "Add to Foreground" action', f'{SHORTCUT_BASE},D,F', 'registeredaction',
    icon=str(ICON_DIR/'foreground.svg'),
    helpText='When an ROI is created, the image processor will attempt to make a new'
             ' component at that location. Right-click and drag to pan.')
  DRAW_ACT_REM    : PrjParam = newParam(
    'Activate "Add to Background" action', f'{SHORTCUT_BASE},D, B', 'registeredaction',
    icon=str(ICON_DIR/'background.svg'),
    helpText='When an ROI is created, the image processor will attempt to take the enclosed'
             ' area away from the current component shape. Right-click and drag to pan.')
  DRAW_ACT_SELECT : PrjParam = newParam(
    'Activate "Select" draw action', f'{SHORTCUT_BASE},D, S', 'registeredaction', icon=str(ICON_DIR / 'select.svg'),
    helpText='When component boundaries are enclosed by this ROI, they will be selected'
             ' in the component table. Right-click and drag to pan.')
  DRAW_ACT_PAN    : PrjParam = newParam(
    'Activate "Pan" draw action', f'{SHORTCUT_BASE},D,P', 'registeredaction', icon=str(ICON_DIR / 'pan.svg'),
    helpText='No ROI will be drawn in this mode. Right- or left-click and drag to pan.')
PRJ_CONSTS = _PrjConsts()
