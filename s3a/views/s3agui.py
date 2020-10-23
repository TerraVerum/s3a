import warnings
from functools import partial
from pathlib import Path
from typing import Optional, Union, Dict, Any

import numpy as np
import pyqtgraph as pg
import qdarkstyle
from pandas import DataFrame as df
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
from pyqtgraph.console import ConsoleWidget

from s3a import plugins, RunOpts
from s3a.constants import LAYOUTS_DIR, FR_CONSTS, REQD_TBL_FIELDS
from s3a.constants import _FREnums, FR_ENUMS
from s3a.graphicsutils import create_addMenuAct, makeExceptionsShowDialogs, \
  autosaveOptsDialog, popupFilePicker, \
  disableAppDuringFunc, saveToFile, dialogGetSaveFileName, addDirItemsToMenu, \
  restoreExceptionBehavior, contextMenuFromEditorActions, ScrollableErrorDialog
from s3a.generalutils import attemptFileLoad
from s3a.models.s3abase import S3ABase
from s3a.parameditors import ParamEditor, ParamEditorDockGrouping, FR_SINGLETON
from s3a.structures import S3AWarning, XYVertices
from s3a.views.buttons import ButtonCollection

__all__ = ['S3A']

@FR_SINGLETON.registerGroup(FR_CONSTS.CLS_ANNOTATOR)
class S3A(S3ABase):
  sigLayoutSaved = QtCore.Signal()
  S3A_INST = None

  @classmethod
  def __initEditorParams__(cls):
    super().__initEditorParams__()
    cls.toolsEditor = ParamEditor.buildClsToolsEditor(cls, 'Main Window')

  def __init__(self, parent=None, guiMode=True, loadLastState=None,
               **quickLoaderArgs):
    # Wait to import quick loader profiles until after self initialization so
    # customized loading functions also get called
    superLoaderArgs = {'author': quickLoaderArgs.pop('author', None)}
    super().__init__(parent, **superLoaderArgs)
    self.toolsEditor.registerFunc(self.estimateBoundaries_gui, btnOpts=FR_CONSTS.TOOL_ESTIMATE_BOUNDARIES)
    self.toolsEditor.registerFunc(self.clearBoundaries, btnOpts=FR_CONSTS.TOOL_CLEAR_BOUNDARIES)
    self.toolsEditor.registerFunc(self.exportCompList_gui, btnOpts=FR_CONSTS.TOOL_EXPORT_COMP_LIST)
    if guiMode:
      warnings.simplefilter('error', S3AWarning)
      makeExceptionsShowDialogs(self)
    def saveRecentLayout(_folderName: Path):
      outFile = _folderName/'savedLayout'
      self.saveLayout(outFile)
      return str(outFile)
    self.appStateEditor.addImportExportOpts('layout', self.loadLayout, saveRecentLayout)
    self.APP_TITLE = 'FICS Semi-Supervised Semantic Annotator'
    self.CUR_COMP_LBL = 'Current Component ID:'
    self.setWindowTitle(self.APP_TITLE)

    self.curCompIdLbl = QtWidgets.QLabel(self.CUR_COMP_LBL)

    # Dummy editor for layout options since it doesn't really have editable settings
    # Maybe later this can be elevated to have more options
    self.layoutEditor = ParamEditor(self, None, LAYOUTS_DIR, 'dockstate', 'Layout')

    self._buildGui()
    self._buildMenu()
    self._hookupSignals()

    self.focusedImg.sigPluginChanged.connect(lambda: self.updateFocusedToolsGrp())
    for plugin in FR_SINGLETON.tableFieldPlugins:
      if isinstance(plugin, plugins.VerticesPlugin):
        # TODO: Config option for which plugin to load by default?
        self.focusedImg.changeCurrentPlugin(plugin)
        break

    # Load layout options
    self.saveLayout('Default', allowOverwriteDefault=True)

    if len(quickLoaderArgs) > 0:
      self.appStateEditor.loadParamState(stateDict=quickLoaderArgs)

    if guiMode:
      QtCore.QTimer.singleShot(0, lambda: self._maybeLoadLastState_gui(loadLastState, quickLoaderArgs))
    elif loadLastState:
      self.loadLastState(quickLoaderArgs)
    # Needs to be reset if loading last state also added new components, but no user changes
    # were made
    self.hasUnsavedChanges = False

  def _hookupSignals(self):
    # Buttons
    self.openImgAct.triggered.connect(lambda: self.setMainImg_gui())
    self.resetTblConfigAct.triggered.connect(lambda: self.resetTblFields_gui())

    FR_SINGLETON.colorScheme.registerFunc(self.updateTheme, FR_CONSTS.CLS_ANNOTATOR.name, runOpts=RunOpts.ON_CHANGED)

    # Menu options
    # FILE
    self.saveLayoutAct.triggered.connect(self.saveLayout_gui)
    self.sigLayoutSaved.connect(self._populateLoadLayoutOptions)

    self.exportCompListAct.triggered.connect(self.exportCompList_gui)
    self.exportLabelImgAct.triggered.connect(self.exportLabeledImg_gui)
    self.loadCompsAct_merge.triggered.connect(lambda: self.loadCompList_gui(FR_ENUMS.COMP_ADD_AS_MERGE))
    self.loadCompsAct_new.triggered.connect(lambda: self.loadCompList_gui(FR_ENUMS.COMP_ADD_AS_NEW))
    self.startAutosaveAct.triggered.connect(self.startAutosave_gui)
    self.stopAutosaveAct.triggered.connect(self.stopAutosave)
    self.userGuideAct.triggered.connect(
      lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl('https://gitlab.com/ficsresearch/s3a/-/wikis/home')))
    self.aboutQtAct.triggered.connect(lambda: QtWidgets.QMessageBox.aboutQt(self, 'About Qt'))

    # EDIT
    stack = FR_SINGLETON.actionStack
    self.undoAct.triggered.connect(lambda: stack.undo())
    self.redoAct.triggered.connect(lambda: stack.redo())
    def updateUndoRedoTxts():
      self.undoAct.setText(f'Undo: {stack.undoDescr}')
      self.redoAct.setText(f'Redo: {stack.redoDescr}')
    stack.stackChangedCallbacks.append(updateUndoRedoTxts)

    # ANALYTICS
    self.modCompAnalyticsAct.triggered.connect(self.showModCompAnalytics)

    # TOOLS
    self.devConsoleAct.triggered.connect(self.showDevConsole)

    self.saveAllEditorDefaults()
    for editor in FR_SINGLETON.registerableEditors:
      editor.setAllExpanded(False)

  def _buildGui(self):
    self.setDockNestingEnabled(True)
    self.setTabPosition(QtCore.Qt.AllDockWidgetAreas, QtWidgets.QTabWidget.North)

    centralwidget = QtWidgets.QWidget(self)
    self.mainImg.setParent(centralwidget)
    layout = QtWidgets.QVBoxLayout(centralwidget)

    self.setCentralWidget(centralwidget)
    layout.addWidget(self.mainImg.drawOptsWidget)
    layout.addWidget(self.mainImg.toolsGrp)
    layout.addWidget(self.mainImg)

    focusedImgDock = QtWidgets.QDockWidget('Focused Image Window', self)
    focusedImgDock.setFeatures(focusedImgDock.DockWidgetMovable|focusedImgDock.DockWidgetFloatable)
    focusedImgContents = QtWidgets.QWidget(self)
    self.focusedImg.setParent(focusedImgContents)
    focusedImgDock.setWidget(focusedImgContents)
    focusedImgDock.setObjectName('Focused Image Dock')
    self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, focusedImgDock)


    focusedLayout = QtWidgets.QVBoxLayout(focusedImgContents)
    focusedLayout.addWidget(self.focusedImg.drawOptsWidget)
    focusedLayout.addWidget(self.focusedImg.toolsGrp)
    focusedLayout.addWidget(self.curCompIdLbl, 0, QtCore.Qt.AlignHCenter)
    focusedLayout.addWidget(self.focusedImg)
    self._focusedLayout = focusedLayout

    sharedMenuWidgets = [self.mainImg, self.compTbl]
    for first, second in zip(sharedMenuWidgets, reversed(sharedMenuWidgets)):
      first.menu.addMenu(contextMenuFromEditorActions(second.toolsEditor, menuParent=first.menu))

    tableDock = QtWidgets.QDockWidget('Component Table Window', self)
    tableDock.setFeatures(tableDock.DockWidgetMovable|tableDock.DockWidgetFloatable)

    tableDock.setObjectName('Component Table Dock')
    tableContents = QtWidgets.QWidget(tableDock)
    tableLayout = QtWidgets.QVBoxLayout(tableContents)
    tableLayout.addWidget(self.compTbl)
    tableDock.setWidget(tableContents)

    self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, tableDock)

    # STATUS BAR
    self.statBar = QtWidgets.QStatusBar(self)
    self.setStatusBar(self.statBar)

    authorName = FR_SINGLETON.tableData.annAuthor
    self.mouseCoords = QtWidgets.QLabel(f"Author: {authorName} Mouse Coords")

    self.pxColor = QtWidgets.QLabel("Pixel Color")

    self.mainImg.sigMousePosChanged.connect(lambda pos, pxColor: self.setInfo(pos, pxColor))
    # self.focusedImg.sigMousePosChanged.connect(lambda info: setInfo(info))
    self.statBar.show()
    self.statBar.addWidget(self.mouseCoords)
    self.statBar.addWidget(self.pxColor)

    # EDITORS
    FR_SINGLETON.sigDocksAdded.connect(lambda newDocks: self._addEditorDocks(newDocks))
    self._addEditorDocks()

  def changeFocusedComp(self, newComps: df, forceKeepLastChange=False):
    ret = super().changeFocusedComp(newComps, forceKeepLastChange)
    self.curCompIdLbl.setText(f'Component ID: {self.focusedImg.compSer[REQD_TBL_FIELDS.INST_ID]}')
    return ret

  def updateFocusedToolsGrp(self):
    newPlugin = self.focusedImg.currentPlugin
    newEditors = [self.focusedImg.toolsEditor]
    if newPlugin is not None:
      newEditors.append(newPlugin.toolsEditor)
    newTools = ButtonCollection.fromToolsEditors(newEditors, self.focusedImg)
    try:
      self._focusedLayout.replaceWidget(self.focusedImg.toolsGrp, newTools)
      self.focusedImg.toolsGrp.deleteLater()
      self.focusedImg.toolsGrp = newTools
    except AttributeError:
      # Fails when window is not yet constructed
      pass
    if len(newTools.paramToBtnMapping) == 0:
      newTools.hide()

  def resetTblFields_gui(self):
    fileDlg = QtWidgets.QFileDialog()
    outFname, _ = fileDlg.getOpenFileName(self, 'Select Table Config File', '',
                                          'All Files (*.*);; Config Files (*.yml)')
    if len(outFname) > 0:
      FR_SINGLETON.tableData.loadCfg(outFname)
      self.resetTblFields()

  def _buildMenu(self):
    # -----
    # MENU BAR
    # -----
    # Top Level
    self.menubar = QtWidgets.QMenuBar(self)
    self.menuFile = QtWidgets.QMenu('&File', self.menubar)
    self.menuEdit = QtWidgets.QMenu('&Edit', self.menubar)
    self.menuAnalytics = QtWidgets.QMenu('&Analytics', self.menubar)
    self.menuHelp = QtWidgets.QMenu('&Help', self.menubar)
    menuTools = QtWidgets.QMenu('&Tools', self.menubar)

    toolbar: QtWidgets.QToolBar = self.addToolBar('Parameter Editors')
    toolbar.setObjectName('Parameter Edtor Toolbar')
    self.paramToolbar = toolbar

    pluginToolbar = self.addToolBar('Plugin Editors')
    pluginToolbar.setObjectName('Plugin Editor Toolbar')
    self.pluginToolbar = pluginToolbar

    self.menubar.addMenu(self.menuFile)
    self.menubar.addMenu(self.menuEdit)
    self.menubar.addMenu(self.menuAnalytics)
    self.menubar.addMenu(menuTools)
    self.menubar.addMenu(self.menuHelp)

    # File / Image
    self.openImgAct = create_addMenuAct(self, self.menuFile, '&Open Image')

    # File / Config
    self.resetTblConfigAct = create_addMenuAct(self, self.menuFile, 'Select &Table Configuration')

    # File / layout
    self.menuLayout = create_addMenuAct(self, self.menuFile, '&Layout', True)
    self.saveLayoutAct = create_addMenuAct(self, self.menuLayout, 'Save Layout')
    self.menuLayout.addSeparator()

    # File / components
    self.menuExport = create_addMenuAct(self, self.menuFile, '&Export...', True)
    self.exportCompListAct = create_addMenuAct(self, self.menuExport, '&Component List')
    self.exportLabelImgAct = create_addMenuAct(self, self.menuExport, '&Labeled Image')

    self.menuLoad_Components = create_addMenuAct(self, self.menuFile, '&Import', True)
    self.loadCompsAct_merge = create_addMenuAct(self, self.menuLoad_Components, 'Update as &Merge')
    self.loadCompsAct_new = create_addMenuAct(self, self.menuLoad_Components, 'Append as &New')

    # File / autosave
    self.menuAutosave = create_addMenuAct(self, self.menuFile, '&Autosave...', True)
    self.startAutosaveAct = create_addMenuAct(self, self.menuAutosave, 'Star&t Autosave')
    self.stopAutosaveAct = create_addMenuAct(self, self.menuAutosave, 'Sto&p Autosave')


    # Edit
    self.undoAct = create_addMenuAct(self, self.menuEdit, '&Undo')
    self.undoAct.setShortcut('Ctrl+Z')
    self.redoAct = create_addMenuAct(self, self.menuEdit, '&Redo')
    self.redoAct.setShortcut('Ctrl+Y')

    # Analytics
    self.modCompAnalyticsAct = create_addMenuAct(self, self.menuAnalytics, 'Modified Component')

    # Tools
    self.devConsoleAct = create_addMenuAct(self, menuTools, 'Show Developer Console')

    # Help
    self.userGuideAct = create_addMenuAct(self, self.menuHelp, 'Online User Guide')
    self.aboutQtAct = create_addMenuAct(self, self.menuHelp, 'Qt Version Info')

    self.setMenuBar(self.menubar)

    pluginDocks = {p.docks: p for p in FR_SINGLETON.plugins}
    # SETTINGS
    for docks in FR_SINGLETON.docks:
      if docks not in pluginDocks:
        self.createMenuOptForDock(docks, parentToolbar=toolbar)

    # This is a bit tricky. If default args are left unfilled, qt slots will fill
    # with 'false's which breaks the function call. However, lambdas can't be used
    # inside a for-loop since the bound variable value won't be correct. To
    # fix this, make a function that generates a function taking no arguments.
    # This ensures (1) bound scope at eval time is correct for lambda and (2)
    # extra args aren't populated with False by qt
    def activator(_plugin):
      def inner():
        self.focusedImg.changeCurrentPlugin(_plugin)
      return inner

    for docks, plugin in pluginDocks.items():
      if docks is None:
        docks = ParamEditorDockGrouping([plugin.toolsEditor], plugin.name)
      menu = self.createMenuOptForDock(docks, parentToolbar=pluginToolbar)
      if plugin in FR_SINGLETON.tableFieldPlugins:
        allActs = menu.actions()
        beforeAct = allActs[0] if len(allActs) > 0 else None
        newAct = QtWidgets.QAction('&Activate', self)
        activatePlugin = partial(self.focusedImg.changeCurrentPlugin, plugin)
        # Need to define separate lambda so that function call forces no args
        newAct.triggered.connect(activator(plugin))
        menu.insertAction(beforeAct, newAct)

  def _maybeLoadLastState_gui(self, loadLastState: bool=None,
                              quickLoaderArgs:dict=None):
    """
    Helper function to determine whether the last application state should be loaded,
    and loads the last state if desired.
    :param loadLastState: If *None*, the user will be prompted via dialog for whether
      to load the last application state. Otherwise, its boolean value is used.
    :param quickLoaderArgs: Additional dict arguments which if supplied will override
      the default options where applicable.
    """
    if not self.appStateEditor.RECENT_STATE_FNAME.exists():
      return
    if loadLastState is None:
      loadLastState = QtWidgets.QMessageBox.question(
        self, 'Load Previous State', 'Do you want to load all previous app'
                                     ' settings (image, annotations, algorithms, etc.)?',
        QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No) == QtWidgets.QMessageBox.Yes
    if loadLastState:
      self.loadLastState_gui(quickLoaderArgs)

  def loadLayout(self, layoutName: Union[str, Path]):
    layoutName = Path(layoutName)
    if not layoutName.is_absolute():
      layoutName = LAYOUTS_DIR/f'{layoutName}.dockstate'
    self.restoreState(attemptFileLoad(layoutName))

  def saveLayout(self, layoutName: Union[str, Path]=None, allowOverwriteDefault=False):
    dockStates = self.saveState().data()
    if Path(layoutName).is_absolute():
      savePathPlusStem = layoutName
    else:
      savePathPlusStem = LAYOUTS_DIR/layoutName
    saveFile = savePathPlusStem.with_suffix(f'.dockstate')
    saveToFile(dockStates, saveFile,
               allowOverwriteDefault=allowOverwriteDefault)
    self.sigLayoutSaved.emit()

  def showDevConsole(self):
    namespace = dict(app=self, rtf=REQD_TBL_FIELDS, singleton=FR_SINGLETON)
    # "dict" default is to use repr instead of string for internal elements, so expanding
    # into string here ensures repr is not used
    nsPrintout = [f"{k}: {v}" for k, v in namespace.items()]
    text = f'Starting console with variables:\n' \
           f'{nsPrintout}'
    console = ConsoleWidget(self, namespace=namespace, text=text)
    console.setWindowFlags(QtCore.Qt.Window)
    console.show()

  def setMainImg_gui(self):
    fileFilter = "Image Files (*.png *.tif *.jpg *.jpeg *.bmp *.jfif);;All files(*.*)"
    fname = popupFilePicker(self, 'Select Main Image', fileFilter)
    if fname is not None:
      with pg.BusyCursor():
        self.setMainImg(fname)

  def startAutosave_gui(self):
    saveDlg = autosaveOptsDialog(self)
    success = saveDlg.exec_()
    if success:
      try:
        interval = saveDlg.intervalEdit.value()
        baseName = saveDlg.baseFileNameEdit.text()
        folderName = Path(saveDlg.folderName)
      except AttributeError:
        warnings.warn('Some information was not provided -- autosave not started.', S3AWarning)
      else:
        self.startAutosave(interval, folderName, baseName)

  def exportCompList_gui(self):
    """Saves the component table to a file"""
    fileDlg = QtWidgets.QFileDialog()
    fileFilters = self.compIo.handledIoTypes_fileFilter('csv', **{'*': 'All Files'})
    outFname, _ = fileDlg.getSaveFileName(self, 'Select Save File', '', fileFilters)
    if len(outFname) > 0:
      super().exportCompList(outFname)

  def exportLabeledImg_gui(self):
    """
    # Note -- These three functions will be a single dialog with options
    # for each requested parameter. It will look like the TableFilterEditor dialog.
    types: List[FRCompParams] = getTypesFromUser()
    outFile = getOutFileFromUser()
    exportLegend = getExpLegendFromUser()
    """
    fileDlg = QtWidgets.QFileDialog()
    # TODO: Delegate this to the exporter. Make a function that makes the right file filter,
    #   and calls the right exporter function after the filename is retrieved.
    fileFilters = self.compIo.handledIoTypes_fileFilter('png', **{'*': 'All Files'})
    fname, _ = fileDlg.getSaveFileName(self, 'Select Save File', '', fileFilters)
    if len(fname) > 0:
      super().exportLabeledImg(fname)

  def loadCompList_gui(self, loadType: _FREnums):
    # TODO: See note about exporting comps. Delegate the filepicker activity to importer
    fileFilter = self.compIo.handledIoTypes_fileFilter(['csv', 'pkl'])
    fname = popupFilePicker(self, 'Select Load File', fileFilter)
    if fname is None:
      return
    self.loadCompList(fname, loadType)

  def saveLayout_gui(self):
    outName = dialogGetSaveFileName(self, 'Layout Name')
    if outName is None or outName == '':
      return
    self.saveLayout(outName)

  # ---------------
  # BUTTON CALLBACKS
  # ---------------
  @disableAppDuringFunc
  def estimateBoundaries_gui(self):
    """
    Estimates component boundaries for the whole image. This is functionally
    equivalent to using a square ROI over the whole image while selecting *New
    component for each separate boundary*=True
    """
    self.estimateBoundaries()

  def loadLastState_gui(self, quickLoaderArgs: dict=None):
    with pg.BusyCursor():
      # TODO: Also show a progress bar, which is the main reason for using a
      #  separate gui function
      self.loadLastState(quickLoaderArgs)

  def loadLastState(self, quickLoaderArgs: dict=None):
    self.appStateEditor.loadParamState(overrideDict=quickLoaderArgs)


  def closeEvent(self, ev: QtGui.QCloseEvent):
    # Confirm all components have been saved
    shouldExit = False
    if self.hasUnsavedChanges:
      ev.ignore()
      if (QtWidgets.QMessageBox().question(self, 'Confirm Exit',
                                         'Component table has unsaved changes.\nAre you sure you want to exit?',
                                         QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Cancel)
          == QtWidgets.QMessageBox.Ok):
        shouldExit = True
    else:
      shouldExit = True
    if shouldExit:
      # Clean up all editor windows, which could potentially be left open
      ev.accept()
      FR_SINGLETON.close()
      restoreExceptionBehavior()
      self.appStateEditor.saveParamState()

  def forceClose(self):
    """
    Allows the app to close even if it has unsaved changes. Useful for closing
    within a script
    """
    self.hasUnsavedChanges = False
    self.close()

  def _addEditorDocks(self, docks=None):
    if docks is None:
      docks = FR_SINGLETON.docks
    # Define out here to retain scope
    def fixDWFactory(_dock):
      # Necessary since defining func in loop will cause problems otherwise
      def doFix(tabIdx):
        self._fixDockWidth(_dock, tabIdx)
      return doFix

    dock = None
    for dock in docks:
      dock.setParent(self)
      self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
      if isinstance(dock, ParamEditorDockGrouping):
        dock.tabs.currentChanged.connect(fixDWFactory(dock))
    for nextEditor in docks[:-1]:
      self.tabifyDockWidget(dock, nextEditor)

  def _createMenuOptForEditor(self, editor: ParamEditor,
                              loadFunc=None, overrideName=None):
    def defaultLoadFunc(objForMenu: ParamEditor, nameToLoad: str) -> Optional[dict]:
      with pg.BusyCursor():
        return objForMenu.loadParamState(nameToLoad)

    if overrideName is None:
      overrideName = editor.name
    if loadFunc is None:
      loadFunc = partial(defaultLoadFunc, editor)
    newMenu = QtWidgets.QMenu(overrideName, self)
    editAct = QtWidgets.QAction('Open ' + overrideName, self)
    newMenu.addAction(editAct)
    newMenu.addSeparator()
    def showFunc(_editor=editor):
      if isinstance(_editor.dock, ParamEditorDockGrouping):
        tabs: QtWidgets.QTabWidget = _editor.dock.tabs
        dockIdx = tabs.indexOf(_editor.dockContentsWidget)
        tabs.setCurrentIndex(dockIdx)
      self._fixDockWidth(_editor.dock)
      _editor.dock.show()
      # "Show" twice forces 1) window to exist and 2) it is currently raised and focused
      # These guarantees are not met if "show" is only called once
      _editor.dock.raise_()
    editAct.triggered.connect(lambda: showFunc())
    populateFunc = partial(self.populateParamEditorMenuOpts, editor, newMenu, loadFunc)
    editor.sigParamStateCreated.connect(populateFunc)
    # Initialize default menus
    populateFunc()
    editor.hasMenuOption = True
    return newMenu

  def _fixDockWidth(self, dock: Union[ParamEditorDockGrouping, ParamEditor], tabIdx: int=None):
    if isinstance(dock, ParamEditorDockGrouping):
      if tabIdx is None:
        tabIdx = dock.tabs.currentIndex()
      curParamEditor = dock.editors[tabIdx]
    else:
      curParamEditor = dock
    curParamEditor.tree.resizeColumnToContents(0)
    minWidth = curParamEditor.width() + 100
    if dock.width() < minWidth:
      self.resizeDocks([dock], [curParamEditor.width()+100], QtCore.Qt.Horizontal)

  def createMenuOptForDock(self,
                           dockEditor: Union[ParamEditor, ParamEditorDockGrouping],
                           loadFunc=None, parentBtn: QtWidgets.QPushButton=None,
                           parentToolbar=None):
    if parentBtn is None:
      parentBtn = QtWidgets.QPushButton()
    if isinstance(dockEditor, ParamEditor):
      parentBtn.setText(dockEditor.name)
      menu = self._createMenuOptForEditor(dockEditor, loadFunc)
      parentBtn.setMenu(menu)
    else:
      # FRParamEditorDockGrouping
      parentBtn.setText(dockEditor.name)
      menu = QtWidgets.QMenu(self)
      parentBtn.setMenu(menu)
      # newMenu = create_addMenuAct(self, parentBtn, dockEditor.name, True)
      for editor in dockEditor.editors:
        # "Main Image Settings" -> "Settings"
        tabName = dockEditor.getTabName(editor)
        nameWithoutBase = tabName
        menu.addMenu(self._createMenuOptForEditor(editor, loadFunc, overrideName=nameWithoutBase))
    if parentToolbar is not None:
      parentToolbar.addWidget(parentBtn)
    return menu

  def _populateLoadLayoutOptions(self):
    layoutGlob = LAYOUTS_DIR.glob('*.dockstate')
    addDirItemsToMenu(self.menuLayout, layoutGlob, self.loadLayout)

  def setInfo(self, xyPos: XYVertices, pxColor: np.ndarray):
    if pxColor is None: return
    authorName = FR_SINGLETON.tableData.annAuthor
    if self.srcImgFname is not None:
      fname = self.srcImgFname.name
    else:
      fname = 'None'

    self.mouseCoords.setText(f'Author: {authorName}'
                             f' | Image: {fname}'
                             f' | Mouse (x,y): {xyPos[0]}, {xyPos[1]}'
                             f' | Pixel Color: ')
    self.pxColor.setText(f'{pxColor}')
    if pxColor.dtype == float:
      # Turn to uint
      pxColor = (pxColor*255).astype('uint8')
    # Regardless of the number of image channels, display as RGBA color
    if pxColor.size == 1:
      # noinspection PyTypeChecker
      pxColor = np.array(pxColor.tolist()*3 + [255])
    elif pxColor.size == 3:
      pxColor = np.concatenate([pxColor, [255]])
    # Else: assume already RGBA
    # Determine text color based on background color
    if np.mean(pxColor) > 127:
      fontColor = 'black'
    else:
      fontColor = 'white'
    self.pxColor.setStyleSheet(f'background:rgba{tuple(pxColor)}; color:{fontColor}')

  def updateTheme(self, useDarkTheme=False):
    """
    :param useDarkTheme:
      title: Use dark theme
    """
    style = ''
    if useDarkTheme:
      style = qdarkstyle.load_stylesheet()
    self.setStyleSheet(style)
    for opts in self.focusedImg.drawOptsWidget, self.mainImg.drawOptsWidget:
      opts.horizWidth = opts.layout().minimumSize().width()

  def add_focusComps(self, newComps: df, addType=FR_ENUMS.COMP_ADD_AS_NEW):
    ret = super().add_focusComps(newComps, addType=addType)
    selection = self.compDisplay.selectRowsById(newComps[REQD_TBL_FIELDS.INST_ID])
    if self.isVisible() and self.compTbl.showOnCreate:
      # For some reason sometimes the actual table selection doesn't propagate in time, so
      # directly forward the selection here
      self.compTbl.setSelectedCellsAs_gui(selection)
    return ret

if __name__ == '__main__':
  import sys
  app = QtWidgets.QApplication([])
  win = S3A()
  win.showMaximized()
  sys.exit(app.exec_())