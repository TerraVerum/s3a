from pathlib import Path

from qtextras import OptionsDict, constants as C, fns

from .structures import ComplexXYVertices

__all__ = [
    "BASE_DIR",
    "MENU_OPTS_DIR",
    "ICON_DIR",
    "QUICK_LOAD_DIR",
    "SCHEMES_DIR",
    "LAYOUTS_DIR",
    "TABLE_DIR",
    "GENERAL_PROPERTIES_DIR",
    "SHORTCUTS_DIR",
    "SHORTCUT_BASE",
    "MAIN_IMAGE_DIR",
    "APP_STATE_DIR",
    "CONFIG_DIR",
    "IMAGE_PROCESSORS_DIR",
    "MULTI_PREDICTIONS_DIR",
    "REQD_TBL_FIELDS",
    "PRJ_CONSTS",
    "PRJ_ENUMS",
    "PROJECT_FILE_TYPE",
    "IO_TEMPLATES_DIR",
    "PROJECT_BASE_TEMPLATE",
]

CODE_DIR = Path(__file__).parent
BASE_DIR = Path.home() / ".s3a"

CONFIG_DIR = CODE_DIR / "config"
ICON_DIR = CODE_DIR / "icons"
MENU_OPTS_DIR = BASE_DIR / "menuopts"
IO_TEMPLATES_DIR = CODE_DIR / "compio/templates"

# -----
# EDITORS
# -----
LAYOUTS_DIR = MENU_OPTS_DIR / "layouts"
QUICK_LOAD_DIR = MENU_OPTS_DIR / "quickloader"
SCHEMES_DIR = MENU_OPTS_DIR / "schemes"
GENERAL_PROPERTIES_DIR = MENU_OPTS_DIR / "genprops"
SHORTCUTS_DIR = MENU_OPTS_DIR / "shortcuts"
MAIN_IMAGE_DIR = MENU_OPTS_DIR / "mainimg"
IMAGE_PROCESSORS_DIR = MENU_OPTS_DIR / "imageproc"
MULTI_PREDICTIONS_DIR = MENU_OPTS_DIR / "multiproc"
TABLE_DIR = MENU_OPTS_DIR / "table"
APP_STATE_DIR = BASE_DIR


PROJECT_BASE_TEMPLATE = CONFIG_DIR / "projectcfg.yml"
PROJECT_FILE_TYPE = "s3aprj"

# Ensure menuopts and layouts directories exist
LAYOUTS_DIR.mkdir(parents=True, exist_ok=True)
APP_STATE_DIR.mkdir(parents=True, exist_ok=True)

SHORTCUT_BASE = "Shift"


class PRJ_ENUMS:
    # --------------------------
    # COMPONENTS
    # --------------------------
    COMPONENT_ADD_AS_MERGE = "merge"
    COMPONENT_ADD_AS_NEW = "new"
    COMPONENT_REMOVE_ALL = None

    # --------------------------
    # REGION CREATION
    # --------------------------
    BUFFER_UNDO = "Undo"
    BUFFER_REDO = "Redo"

    # --------------------------
    # VERTICES
    # --------------------------
    HIERARCHY_ALL_FILLED = "All Filled"

    # --------------------------
    # COMMON DATAFRAME FIELDS
    # --------------------------
    FIELD_SELECTED = "selected"
    FIELD_FOCUSED = "focused"
    FIELD_LABEL = "label"

    # --------------------------
    # I/O SPECIFICATIONS
    # --------------------------
    IO_EXPORT = "export"
    IO_IMPORT = "import"
    IO_ROUND_TRIP = "round trip"
    IO_META_MAPPING = "mapping"
    IO_META_FIELD = "field"

    # --------------------------
    # LOGGING / FEEDBACK CAPABILITIES
    # --------------------------
    LOG_GUI = "gui"
    LOG_TERM = "term"
    LOG_NONE = "none"
    LOG_FILE = "file"
    LOG_LEVEL_ATTENTION = C.PrjEnums.LOG_LVL_ATTN

    # --------------------------
    # PROCESSING
    # --------------------------
    PROCESS_ADD_TOP = "top"
    PROCESS_ADD_PRIMITIVE = "primitive"
    PROCESS_NO_ADD = None

    HISTORY_UNSPECIFIED = 0
    HISTORY_BACKGROUND = 1
    HISTORY_FOREGROUND = 2

    ROTATION_OPTIMAL = None


class _ReqdTableFields:
    _extraRequired = []

    def __init__(self):
        self._iterFields = []

        # Can't use IOTemplateManager without creating cyclic reference
        fields = fns.attemptFileLoad(IO_TEMPLATES_DIR / "s3a.tblcfg")["fields"]

        ctorItems = [{"name": kk, **vv} for kk, vv in fields.items()]
        ii = 0

        def constructNext():
            nonlocal ii
            ret = OptionsDict(**ctorItems[ii])
            ii += 1
            self._iterFields.append(ret)
            return ret

        self.ID = constructNext()
        self.IMAGE_FILE = constructNext()
        self.VERTICES = constructNext()
        # Special case: Vertices is non-primitive type
        self.VERTICES.value = ComplexXYVertices()

    def addField(self, field: OptionsDict):
        if field not in self:
            self._extraRequired.append(field)

    def removeField(self, field: OptionsDict):
        if field in self._extraRequired:
            self._extraRequired.remove(field)

    def __iter__(self):
        for lst in self._iterFields, self._extraRequired:
            yield from lst.__iter__()


REQD_TBL_FIELDS = _ReqdTableFields()


class PRJ_CONSTS:
    # --------------------------
    # SCHEME PARAMETERS
    # --------------------------
    SCHEME_USE_DARK_THEME = OptionsDict("Use dark theme", False)
    SCHEME_BOUND_CLR = OptionsDict("Normal Boundary Color", "#ff0", "color")
    SCHEME_SEL_BOUND_CLR = OptionsDict("Selected Boundary Color", "#00f", "color")
    SCHEME_FOC_BRUSH_CLR = OptionsDict("Focused Brush Color", "#f007", "color")
    SCHEME_BOUND_WIDTH = OptionsDict("Boundary Width", 7)
    SCHEME_REG_VERT_COLOR = OptionsDict("Vertex Color", "#0f0", "color")
    SCHEME_REG_FILL_COLOR = OptionsDict("Fill Color", "#00ff0046", "color")
    SCHEME_ROI_LINE_CLR = OptionsDict("ROI Line Color", "#fff", "color")
    SCHEME_ROI_LINE_WIDTH = OptionsDict("ROI Line Width", 1)
    SCHEME_GRID_LINE_WIDTH = OptionsDict("Grid Line Width", 1)
    SCHEME_GRID_CLR = OptionsDict("Grid Line Color", "#fff", "color")
    SCHEME_SHOW_GRID = OptionsDict("Show Grid", False)
    SCHEME_LBL_COL = OptionsDict(
        "Labeling Column", REQD_TBL_FIELDS.ID.name, type="list", limits=[]
    )

    # --------------------------
    # REGION-CREATION PARAMETERS
    # --------------------------
    PROP_MIN_COMP_SZ = OptionsDict("Minimum New Component Size (px)", 50)
    PROP_UNDO_BUF_SZ = OptionsDict("Undo Buffer Size", 300)
    PROP_REG_APPROX_EPS = OptionsDict(
        "Region Simplification Tolerance", 1.0, dec=True, limits=[-1, None]
    )
    PROP_COMP_SEL_BHV = OptionsDict(
        "Component Selection Behavior",
        "Entire Component",
        "list",
        limits={"Boundary Only": True, "Entire Component": False},
        helpText="When `Boundary Only`, components are selected when clicking on "
        "their boundaries.\nWhen `Entire Component`, clicking anywhere within "
        "the component will select it",
    )
    PROP_FIELD_INFO_ON_SEL = OptionsDict(
        "Show field info after selection",
        False,
        helpText="Whether to show field information in the main image"
        " every time the current selection changes",
    )
    PROP_VERT_SORT_BHV = OptionsDict(
        "Vertices Sorting",
        "X First",
        "list",
        limits=["X First", "Y First"],
        helpText="Whether to sort first by X or Y when sorting a vertices-like field",
    )
    PROP_SHOW_TBL_ON_COMP_CREATE = OptionsDict(
        "Show popup table when creating component", False
    )
    PROP_SCALE_PEN_WIDTH = OptionsDict("Scale pen width to image pixel size", True)

    PROP_COLLECT_USR_METRICS = OptionsDict(
        "Collect user metrics",
        False,
        helpText="Collect user metrics for smarter segmentation algorithms",
    )
    # --------------------------
    # MISC TOOLS
    # --------------------------
    TOOL_UNDO = OptionsDict("Undo", "Ctrl+Z")
    TOOL_REDO = OptionsDict("Redo", "Ctrl+Y")
    # --------------------------
    # IMAGE TOOLS
    # --------------------------
    TOOL_MERGE_COMPS = OptionsDict(
        "Merge Selected", f"{SHORTCUT_BASE}+S,M", icon=str(ICON_DIR / "merge.svg")
    )
    TOOL_SPLIT_COMPS = OptionsDict(
        "Split Selected", f"{SHORTCUT_BASE}+S,S", icon=str(ICON_DIR / "split.svg")
    )
    TOOL_REM_OVERLAP = OptionsDict(
        "Remove Component Overlap",
        f"{SHORTCUT_BASE}+S,E",
        icon=str(ICON_DIR / "mutex.svg"),
    )
    TOOL_COPY_REGIONS = OptionsDict(
        "Copy Selected", f"{SHORTCUT_BASE}+S,C", icon=str(ICON_DIR / "copy.svg")
    )
    TOOL_MOVE_REGIONS = OptionsDict(
        "Move Selected", f"{SHORTCUT_BASE}+S,V", icon=str(ICON_DIR / "move.svg")
    )
    TOOL_CLEAR_FOC_REGION = OptionsDict(
        "Clear", f"{SHORTCUT_BASE}+V,C", icon=str(ICON_DIR / "clear.svg")
    )
    TOOL_RESET_FOC_REGION = OptionsDict(
        "Reset", f"{SHORTCUT_BASE}+V,R", icon=str(ICON_DIR / "reset.svg")
    )
    TOOL_FILL_FOC_REGION = OptionsDict(
        "Fill", f"{SHORTCUT_BASE}+V,F", icon=str(ICON_DIR / "fill.svg")
    )
    TOOL_INVERT_FOC_REGION = OptionsDict(
        "Invert", f"{SHORTCUT_BASE}+V,I", icon=str(ICON_DIR / "invert.svg")
    )
    TOOL_ACCEPT_FOC_REGION = OptionsDict(
        "Accept", "Ctrl+Shift+A", icon=str(ICON_DIR / "accept.svg")
    )
    TOOL_CLEAR_ROI = OptionsDict("Clear ROI", "Esc")
    TOOL_CLEAR_HISTORY = OptionsDict(
        "Clear Processor History",
        f"{SHORTCUT_BASE}+V,H",
        icon=str(ICON_DIR / "clear_history.svg"),
    )
    TOOL_PROC_ANALYTICS = OptionsDict(
        "Show Analytics", icon=str(ICON_DIR / "analytics.svg")
    )
    TOOL_RESET_ZOOM = OptionsDict(
        "Reset Zoom", f"{SHORTCUT_BASE}+Z,R", icon=str(ICON_DIR / "reset_zoom.svg")
    )

    # --------------------------
    # WINDOW TOOLS
    # --------------------------
    TOOL_ESTIMATE_BOUNDARIES = OptionsDict("Estimate Boundaries", "Ctrl+Alt+Shift+E")
    TOOL_CLEAR_BOUNDARIES = OptionsDict("Clear Boundaries", "Ctrl+Alt+Shift+C")
    TOOL_EXPORT_COMP_LIST = OptionsDict("Export Current Table", f"{SHORTCUT_BASE}+E,T")
    TOOL_TBL_SET_SAME_AS_FIRST = OptionsDict("Set Cells as First", "Ctrl+D")
    TOOL_TBL_SET_AS = OptionsDict("Set Cells As...", "Ctrl+Shift+D")
    TOOL_TBL_DEL_ROWS = OptionsDict("Delete Table Rows", "Del")
    TOOL_TBL_ZOOM_TO_COMPS = OptionsDict("Zoom to Selection  ", f"{SHORTCUT_BASE}+Z,S")

    # --------------------------
    # PROJECT
    # --------------------------
    TOOL_PROJ_SAVE = OptionsDict("Save", "Ctrl+S")
    TOOL_PROJ_OPEN = OptionsDict("Open Project", f"{SHORTCUT_BASE}+P,O")
    TOOL_PROJ_OPEN_IMG = OptionsDict("Open Project Image", f"{SHORTCUT_BASE}+I,O")
    TOOL_PROJ_CREATE = OptionsDict("Create Project", f"{SHORTCUT_BASE}+P,C")
    TOOL_PROJ_ADD_IMG = OptionsDict("Add New Image", f"{SHORTCUT_BASE}+I,A")
    TOOL_PROJ_ADD_ANN = OptionsDict("Add New Annotation", f"{SHORTCUT_BASE}+A,A")
    TOOL_PROJ_SETTINGS = OptionsDict("Project Settings...", f"{SHORTCUT_BASE}+P,S")
    TOOL_PROJ_EXPORT = OptionsDict("Export...", f"{SHORTCUT_BASE}+P,E")
    TOOL_AUTOSAVE = OptionsDict("Autosave...", f"{SHORTCUT_BASE}+A,O")

    # --------------------------
    # GLOBAL PREDICTIONS
    # --------------------------
    TOOL_MULT_PRED = OptionsDict(
        "Make Multi-Prediction",
        f"{SHORTCUT_BASE}+M,P",
        icon=str(ICON_DIR / "predict.svg"),
    )

    # --------------------------
    # COMPONENT EXPORT PARAMETERS
    # --------------------------
    EXP_ONLY_VISIBLE = OptionsDict(
        "Only Export Visible Components",
        False,
        helpText="If *True*, only components showing on the main image will be included "
        "in file exports.",
    )
    INCLUDE_FNAME_PATH = OptionsDict(
        "Include full image path on export",
        False,
        None,
        "If *False*, only the image name is included in export information, e.g. "
        "``<img>.png``. Otherwise, the full filepath is included, e.g. "
        "``C:/Users/.../<img>.png``",
    )

    # -------------------
    # DRAWING
    # -------------------
    # Modes
    DRAW_MODE_FOCUSED = OptionsDict(
        'Activate "Edit" draw mode',
        f"{SHORTCUT_BASE}+D,E",
        "registeredaction",
        icon=str(ICON_DIR / "edit.svg"),
    )

    # Shapes
    DRAW_SHAPE_RECT = OptionsDict(
        'Activate "Rectangular" draw shape',
        f"{SHORTCUT_BASE}+D,R",
        "registeredaction",
        icon=str(ICON_DIR / "rectangle.svg"),
    )
    DRAW_SHAPE_POLY = OptionsDict(
        'Activate "Polygon" draw shape',
        f"{SHORTCUT_BASE}+D,Y",
        "registeredaction",
        icon=str(ICON_DIR / "polygon.svg"),
    )
    DRAW_SHAPE_ELLIPSE = OptionsDict(
        'Activate "Ellipse" draw shape',
        f"{SHORTCUT_BASE}+D,L",
        "registeredaction",
        icon=str(ICON_DIR / "ellipse.svg"),
    )
    DRAW_SHAPE_FREE = OptionsDict(
        'Activate "Freehand" draw shape',
        f"{SHORTCUT_BASE}+D,H",
        icon=str(ICON_DIR / "freehand.svg"),
    )
    DRAW_SHAPE_POINT = OptionsDict(
        'Activate "Point" draw shape',
        f"{SHORTCUT_BASE}+D,N",
        icon=str(ICON_DIR / "point.svg"),
    )
    DRAW_SHAPE_NONE = OptionsDict("None")

    # Actions
    DRAW_ACT_CREATE = OptionsDict(
        'Activate "Create Component" action',
        f"{SHORTCUT_BASE}+D,C",
        "registeredaction",
        icon=str(ICON_DIR / "create.svg"),
        helpText="When an ROI is created, the image processor will attempt to make a new"
        " component at that location. Right-click and drag to pan.",
    )
    DRAW_ACT_ADD = OptionsDict(
        'Activate "Add to Foreground" action',
        f"{SHORTCUT_BASE}+D,F",
        "registeredaction",
        icon=str(ICON_DIR / "foreground.svg"),
        helpText="When an ROI is created, the image processor will attempt to make a new"
        " component at that location. Right-click and drag to pan.",
    )
    DRAW_ACT_REM = OptionsDict(
        'Activate "Add to Background" action',
        f"{SHORTCUT_BASE}+D, B",
        "registeredaction",
        icon=str(ICON_DIR / "background.svg"),
        helpText="When an ROI is created, the image processor will attempt to take the "
        "enclosedarea away from the current component shape. Right-click and "
        "drag to pan.",
    )
    DRAW_ACT_SELECT = OptionsDict(
        'Activate "Select" draw action',
        f"{SHORTCUT_BASE}+D, S",
        "registeredaction",
        icon=str(ICON_DIR / "select.svg"),
        helpText="When component boundaries are enclosed by this ROI, they will be "
        "selected in the component table. Right-click and drag to pan.",
    )
    DRAW_ACT_PAN = OptionsDict(
        'Activate "Pan" draw action',
        f"{SHORTCUT_BASE}+D,P",
        "registeredaction",
        icon=str(ICON_DIR / "pan.svg"),
        helpText="No ROI will be drawn in this mode. Right- or left-click and drag "
        "to pan.",
    )
