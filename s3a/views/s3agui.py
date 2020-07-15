import warnings
from functools import partial
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any, List

import numpy as np
import qdarkstyle
from pandas import DataFrame as df
import pyqtgraph as pg
from pyqtgraph.console import ConsoleWidget
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui

from s3a.views.parameditors import FRParamEditor, FRParamEditorDockGrouping, FR_SINGLETON
from s3a.models.s3abase import S3ABase
from s3a.projectvars import LAYOUTS_DIR, FR_CONSTS, FR_ENUMS, APP_STATE_DIR, \
  REQD_TBL_FIELDS
from s3a.projectvars.enums import _FREnums
from s3a.structures import FRS3AWarning, FRVertices
from s3a.graphicsutils import create_addMenuAct, makeExceptionsShowDialogs, \
  autosaveOptsDialog, attemptFileLoad, popupFilePicker, \
  disableAppDuringFunc, saveToFile, dialogGetSaveFileName, addDirItemsToMenu, \
  restoreExceptionBehavior
from .imageareas import FRFocusedImage

__all__ = ['S3A']

@FR_SINGLETON.registerGroup(FR_CONSTS.CLS_ANNOTATOR)
class S3A(S3ABase):
  sigLayoutSaved = QtCore.Signal()

  @classmethod
  def __initEditorParams__(cls):
    super().__initEditorParams__()

  def __init__(self, parent=None, guiMode=True, loadLastState=None,
               **quickLoaderArgs):
    # Wait to import quick loader profiles until after self initialization so
    # customized loading functions also get called
    superLoaderArgs = {'author': quickLoaderArgs.pop('author', None)}
    super().__init__(parent, **superLoaderArgs)
    if guiMode:
      warnings.simplefilter('error', FRS3AWarning)
      makeExceptionsShowDialogs(self)
    def saveRecentLayout():
      outFile = self.appStateEditor.saveDir/'savedLayout'
      self.saveLayout(outFile)
      return str(outFile)
    self.appStateEditor.addImportExportOpts('layout', self.loadLayout, saveRecentLayout)
    self.APP_TITLE = 'FICS Semi-Supervised Semantic Annotator'
    self.CUR_COMP_LBL = 'Current Component ID:'
    self.setWindowTitle(self.APP_TITLE)

    self.focusedImg = FRFocusedImage()
    self.curCompIdLbl = QtWidgets.QLabel(self.CUR_COMP_LBL)
    self.clearRegionBtn = QtWidgets.QPushButton('Clear')
    self.resetRegionBtn = QtWidgets.QPushButton('Reset')
    self.acceptRegionBtn = QtWidgets.QPushButton('Accept')
    self.acceptRegionBtn.setStyleSheet("background-color:lightgreen")

    # Dummy editor for layout options since it doesn't really have editable settings
    # Maybe later this can be elevated to have more options
    self.layoutEditor = FRParamEditor(self, None, LAYOUTS_DIR, 'dockstate', 'Layout')

    self._buildGui()
    self._buildMenu()
    self._hookupSignals()

    # Load layout options
    self.saveLayout('Default', allowOverwriteDefault=True)

    if len(quickLoaderArgs) > 0:
      self.appStateEditor.loadParamState(stateDict=quickLoaderArgs)

    QtCore.QTimer.singleShot(0, lambda: self._maybeLoadLastState(guiMode, loadLastState))

  def _hookupSignals(self):
    # Buttons
    self.openImgAct.triggered.connect(lambda: self.resetMainImg_gui())
    self.clearRegionBtn.clicked.connect(self.clearFocusedRegion)
    self.resetRegionBtn.clicked.connect(lambda: self.resetFocusedRegion())
    self.acceptRegionBtn.clicked.connect(lambda: self.acceptFocusedRegion())

    FR_SINGLETON.scheme.sigParamStateUpdated.connect(self.updateTheme)

    # Menu options
    # FILE
    self.saveLayoutAct.triggered.connect(self.saveLayout_gui)
    self.sigLayoutSaved.connect(self.populateLoadLayoutOptions)

    self.exportCompListAct.triggered.connect(self.exportCompList_gui)
    self.exportLabelImgAct.triggered.connect(self.exportLabeledImg_gui)
    self.loadCompsAct_merge.triggered.connect(lambda: self.loadCompList_gui(FR_ENUMS.COMP_ADD_AS_MERGE))
    self.loadCompsAct_new.triggered.connect(lambda: self.loadCompList_gui(FR_ENUMS.COMP_ADD_AS_NEW))
    self.startAutosaveAct.triggered.connect(self.startAutosave_gui)
    self.stopAutosaveAct.triggered.connect(self.stopAutosave)

    # EDIT
    stack = FR_SINGLETON.actionStack
    self.undoAct.triggered.connect(lambda: stack.undo())
    self.redoAct.triggered.connect(lambda: stack.redo())
    def updateUndoRedoTxts():
      self.undoAct.setText(f'Undo: {stack.undoDescr}')
      self.redoAct.setText(f'Redo: {stack.redoDescr}')
    stack.stackChangedCallbacks.append(updateUndoRedoTxts)

    # ANALYTICS
    self.newCompAnalyticsAct.triggered.connect(self.showNewCompAnalytics)
    self.modCompAnalyticsAct.triggered.connect(self.showModCompAnalytics)

    self.saveAllEditorDefaults()

  def _buildGui(self):
    self.setDockNestingEnabled(True)
    self.setTabPosition(QtCore.Qt.AllDockWidgetAreas, QtWidgets.QTabWidget.North)

    centralwidget = QtWidgets.QWidget(self)
    self.mainImg.setParent(centralwidget)
    layout = QtWidgets.QVBoxLayout(centralwidget)

    self.setCentralWidget(centralwidget)
    layout.addWidget(self.mainImg.drawOptsWidget)
    layout.addWidget(self.mainImg)

    regionBtnLayout = QtWidgets.QHBoxLayout()
    regionBtnLayout.addWidget(self.clearRegionBtn)
    regionBtnLayout.addWidget(self.resetRegionBtn)
    regionBtnLayout.addWidget(self.acceptRegionBtn)

    focusedImgDock = QtWidgets.QDockWidget('Focused Image Window', self)
    focusedImgContents = QtWidgets.QWidget(self)
    self.focusedImg.setParent(focusedImgContents)
    focusedImgDock.setWidget(focusedImgContents)
    focusedImgDock.setObjectName('Focused Image Dock')
    self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, focusedImgDock)


    focusedLayout = QtWidgets.QVBoxLayout(focusedImgContents)
    focusedLayout.addWidget(self.focusedImg.drawOptsWidget)
    focusedLayout.addWidget(self.curCompIdLbl, 0, QtCore.Qt.AlignHCenter)
    focusedLayout.addWidget(self.focusedImg)
    for btn in self.clearRegionBtn, self.resetRegionBtn, self.acceptRegionBtn:
      btn.setParent(focusedImgContents)
    focusedLayout.addLayout(regionBtnLayout)

    tableDock = QtWidgets.QDockWidget('Component Table Window', self)
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

  def _buildMenu(self):
    # -----
    # MENU BAR
    # -----
    # Top Level
    self.menubar = QtWidgets.QMenuBar(self)
    self.menuFile = QtWidgets.QMenu('&File', self.menubar)
    self.menuEdit = QtWidgets.QMenu('&Edit', self.menubar)
    self.menuAnalytics = QtWidgets.QMenu('&Analytics', self.menubar)

    toolbar = self.addToolBar('Parameter Editors')
    toolbar.setObjectName('Parameter Edtor Toolbar')
    self.paramTools = QtWidgets.QMenuBar()
    toolbar.addWidget(self.paramTools)

    self.menubar.addMenu(self.menuFile)
    self.menubar.addMenu(self.menuEdit)
    self.menubar.addMenu(self.menuAnalytics)

    # File / Image
    self.openImgAct = create_addMenuAct(self, self.menuFile, '&Open Image')

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
    self.newCompAnalyticsAct = create_addMenuAct(self, self.menuAnalytics, 'Newest Added Component')
    self.modCompAnalyticsAct = create_addMenuAct(self, self.menuAnalytics, 'Modified Component')

    self.setMenuBar(self.menubar)

    # SETTINGS
    for dock in FR_SINGLETON.docks:
      self.createMenuOptForDock(self.paramTools, dock)
    self.createMenuOptForEditor(self.menuFile, FR_SINGLETON.quickLoader)

  def _maybeLoadLastState(self, guiMode: bool, loadLastState: bool=None):
    """
    Helper function to determine whether the last application state should be loaded,
    and loads the last state if desired.
    :param guiMode: Whether this application is running in gui mode
    :param loadLastState: If *None*, the user will be prompted via dialog for whether
      to load the last application state. Otherwise, its boolean value is used.
    """
    if not self.appStateEditor.RECENT_STATE_FNAME.exists():
      return
    if loadLastState is None and guiMode:
      loadLastState = QtWidgets.QMessageBox.question(
        self, 'Load Previous State', 'Do you want to load all previous app'
                                     ' settings (image, annotations, algorithms, etc.)?',
        QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No) == QtWidgets.QMessageBox.Yes
    if loadLastState and guiMode:
      self.loadLastState_gui()
    elif loadLastState:
      self.loadLastState()

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

  def resetMainImg_gui(self):
    fileFilter = "Image Files (*.png; *.tif; *.jpg; *.jpeg; *.bmp; *.jfif);; All files(*.*)"
    fname = popupFilePicker(self, 'Select Main Image', fileFilter)
    if fname is not None:
      with BusyCursor():
        self.setMainImg(fname)

  def startAutosave_gui(self):
    saveDlg = autosaveOptsDialog(self)
    success = saveDlg.exec()
    if success:
      try:
        interval = saveDlg.intervalEdit.value()
        baseName = saveDlg.baseFileNameEdit.text()
        folderName = Path(saveDlg.folderName)
      except AttributeError:
        warnings.warn('Some information was not provided -- autosave not started.', FRS3AWarning)
      else:
        self.startAutosave(interval, folderName, baseName)

  @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_EXPORT_COMP_LIST)
  def exportCompList_gui(self):
    fileDlg = QtWidgets.QFileDialog()
    # TODO: Delegate this to the exporter. Make a function that makes the right file filter,
    #   and calls the right exporter function after the filename is retrieved.
    fileFilter = self.compIo.handledIoTypes_fileFilter()
    outFname, _ = fileDlg.getSaveFileName(self, 'Select Save File', '', fileFilter)
    if len(outFname) > 0:
      super().exportCompList(outFname)

  def exportLabeledImg_gui(self):
    """
    # Note -- These three functions will be a single dialog with options
    # for each requested parameter. It will look like the FRTableFilterEditor dialog.
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
  @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_ESTIMATE_BOUNDARIES)
  @disableAppDuringFunc
  def estimateBoundaries_gui(self):
    self.estimateBoundaries()

  @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_CLEAR_BOUNDARIES)
  def clearBoundaries_gui(self):
    self.clearBoundaries()

  def loadLastState_gui(self):
    with pg.BusyCursor():
      self.loadLastState()

  def loadLastState(self):
    with FR_SINGLETON.actionStack.group('Load Last Application State'):
      self.appStateEditor.loadParamState()

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
    dock = None
    for dock in docks:
      dock.setParent(self)
      self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
    for nextEditor in docks[:-1]:
      self.tabifyDockWidget(dock, nextEditor)

  def createMenuOptForEditor(self, parentMenu: QtWidgets.QMenu, editor: FRParamEditor,
                             loadFunc=None, overrideName=None):
    def defaultLoadFunc(objForMenu: FRParamEditor, nameToLoad: str) -> Optional[dict]:
      with pg.BusyCursor():
        return objForMenu.loadParamState(nameToLoad)

    if overrideName is None:
      overrideName = editor.name
    if editor.hasMenuOption:
      return
    if loadFunc is None:
      loadFunc = partial(defaultLoadFunc, editor)
    newMenu = QtWidgets.QMenu(overrideName, self)
    editAct = QtWidgets.QAction('Open ' + overrideName, self)
    newMenu.addAction(editAct)
    newMenu.addSeparator()
    def showFunc(_editor=editor):
      _editor.dock.show()
      # "Show" twice forces 1) window to exist and 2) it is currently raised and focused
      # These guarantees are not met if "show" is only called once
      _editor.dock.raise_()
      if isinstance(_editor.dock, FRParamEditorDockGrouping):
        tabs: QtWidgets.QTabWidget = _editor.dock.tabs
        dockIdx = tabs.indexOf(_editor.dockContentsWidget)
        tabs.setCurrentIndex(dockIdx)
    editAct.triggered.connect(lambda: showFunc())
    populateFunc = partial(self.populateParamEditorMenuOpts, editor, newMenu, loadFunc)
    editor.sigParamStateCreated.connect(populateFunc)
    # Initialize default menus
    populateFunc()
    parentMenu.addMenu(newMenu)
    editor.hasMenuOption = True

  def createMenuOptForDock(self, parentMenu: QtWidgets.QMenu,
                           dockEditor: Union[FRParamEditor, FRParamEditorDockGrouping],
                           loadFunc=None):
    if isinstance(dockEditor, FRParamEditor):
      self.createMenuOptForEditor(parentMenu, dockEditor, loadFunc)
    else:
      # FRParamEditorDockGrouping
      newMenu = create_addMenuAct(self, parentMenu, dockEditor.name, True)
      for editor in dockEditor.editors:
        # "Main Image Settings" -> "Settings"
        nameWithoutBase = editor.name.split(dockEditor.name)[1][1:]
        self.createMenuOptForEditor(newMenu, editor, loadFunc, overrideName=nameWithoutBase)

  def populateLoadLayoutOptions(self):
    layoutGlob = LAYOUTS_DIR.glob('*.dockstate')
    addDirItemsToMenu(self.menuLayout, layoutGlob, self.loadLayout)

  def setInfo(self, xyPos: FRVertices, pxColor: np.ndarray):
    if pxColor is None: return
    authorName = FR_SINGLETON.tableData.annAuthor
    self.mouseCoords.setText(f'Author: {authorName} | Mouse (x,y): {xyPos[0]}, {xyPos[1]} | Pixel Color: ')
    self.pxColor.setText(f'{pxColor}')
    if pxColor.dtype == float:
      # Turn to uint
      pxColor = (pxColor*255).astype('uint8')
    # Regardless of the number of image channels, display as RGBA color
    if pxColor.size == 1:
      pxColor = np.array([pxColor[0]]*3 + [255])
    elif pxColor.size == 3:
      pxColor = np.concatenate([pxColor, [255]])
    # Else: assume already RGBA
    # Determine text color based on background color
    if np.mean(pxColor) > 127:
      fontColor = 'black'
    else:
      fontColor = 'white'
    self.pxColor.setStyleSheet(f'background:rgba{tuple(pxColor)}; color:{fontColor}; font-weight: 16px')

  def updateTheme(self, _newScheme: Dict[str, Any]):
    style = ''
    if self.useDarkTheme:
      style = qdarkstyle.load_stylesheet()
    self.setStyleSheet(style)
    for opts in self.focusedImg.drawOptsWidget, self.mainImg.drawOptsWidget:
      opts.horizWidth = opts.layout().minimumSize().width()

  def add_focusComp(self, newComps: df):
    ret = super().add_focusComp(newComps)
    if self.isVisible() and self.compTbl.showOnCreate:
      self.compTbl.setSelectedCellsAs_gui(overrideIds=newComps.index)
    return ret


if __name__ == '__main__':
  import sys
  app = QtWidgets.QApplication([])
  win = S3A()
  win.showMaximized()
  sys.exit(app.exec_())