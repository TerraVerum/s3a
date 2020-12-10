import warnings
from functools import partial
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pyqtgraph as pg
import qdarkstyle
from pandas import DataFrame as df
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
from pyqtgraph.console import ConsoleWidget

import s3a.plugins.tablefield
from s3a import plugins, RunOpts
from s3a.constants import LAYOUTS_DIR, FR_CONSTS as FRC, REQD_TBL_FIELDS
from s3a.constants import _FREnums, FR_ENUMS
from s3a.generalutils import attemptFileLoad
from s3a.graphicsutils import create_addMenuAct, makeExceptionsShowDialogs, \
  autosaveOptsDialog, popupFilePicker, \
  disableAppDuringFunc, saveToFile, dialogGetSaveFileName, addDirItemsToMenu, \
  restoreExceptionBehavior, menuFromEditorActions
from s3a.models.s3abase import S3ABase
from s3a.parameditors import ParamEditor, ParamEditorDockGrouping, FR_SINGLETON, \
  ParamEditorPlugin, TableFieldPlugin
from s3a.plugins import MainImagePlugin, CompTablePlugin, ProjectsPlugin, \
  MiscFunctionsPlugin
from s3a.structures import S3AWarning, XYVertices, FilePath, NChanImg
from s3a.views.buttons import ButtonCollection

__all__ = ['S3A']

_MENU_PLUGINS = [ProjectsPlugin, MiscFunctionsPlugin]

@FR_SINGLETON.registerGroup(FRC.CLS_ANNOTATOR)
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
    # Create toolbars here so plugins instantiated in super init don't throw errors
    self.generalToolbar = QtWidgets.QToolBar('General Plugins')
    self.tblFieldToolbar = QtWidgets.QToolBar('Table Field Plugins')

    super().__init__(parent, **superLoaderArgs)
    for func, param in zip(
        [self.estimateBoundaries_gui, self.clearBoundaries, self.exportAnnotations_gui],
        [FRC.TOOL_ESTIMATE_BOUNDARIES, FRC.TOOL_CLEAR_BOUNDARIES, FRC.TOOL_EXPORT_COMP_LIST]):
      param.opts['ownerObj'] = self
      self.toolsEditor.registerFunc(func, btnOpts=param)
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
    # TODO: Config option for which plugin to load by default?
    self.focusedImg.changeCurrentPlugin(FR_SINGLETON.clsToPluginMapping[
                                          s3a.plugins.tablefield.VerticesPlugin])

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
    self.openAnnsAct.triggered.connect(self.openAnnotation_gui)
    self.exportAnnsAct.triggered.connect(self.exportAnnotations_gui)

    FR_SINGLETON.colorScheme.registerFunc(self.updateTheme, FRC.CLS_ANNOTATOR.name, runOpts=RunOpts.ON_CHANGED)

    # Menu options
    # FILE
    self.saveLayoutAct.triggered.connect(self.saveLayout_gui)
    self.sigLayoutSaved.connect(self._populateLoadLayoutOptions)

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

    self.saveAllEditorDefaults()
    for editor in FR_SINGLETON.registerableEditors:
      editor.setAllExpanded(False)

  def _buildGui(self):
    self.setDockNestingEnabled(True)
    self.setTabPosition(QtCore.Qt.AllDockWidgetAreas, QtWidgets.QTabWidget.North)

    self.tblFieldToolbar.setObjectName('Table Field Plugins')
    self.generalToolbar.setObjectName('General Plugins')
    self.addToolBar(self.generalToolbar)
    self.addToolBar(self.tblFieldToolbar)

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

    plugins = [FR_SINGLETON.clsToPluginMapping[c] for c in [MainImagePlugin, CompTablePlugin]]
    parents = [self.mainImg, self.compTbl]
    for plugin, parent in zip(plugins, reversed(parents)):
      parent.menu.addMenu(menuFromEditorActions([plugin.toolsEditor], plugin.name, menuParent=parent))


    tableDock = QtWidgets.QDockWidget('Component Table Window', self)
    tableDock.setFeatures(tableDock.DockWidgetMovable|tableDock.DockWidgetFloatable)

    tableDock.setObjectName('Component Table Dock')
    tableContents = QtWidgets.QWidget(tableDock)
    tableLayout = QtWidgets.QVBoxLayout(tableContents)
    tableLayout.addWidget(self.compTbl)
    tableDock.setWidget(tableContents)

    self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, tableDock)

    # STATUS BAR
    self.setStatusBar(self.statBar)

    authorName = FR_SINGLETON.tableData.annAuthor
    self.mouseCoords = QtWidgets.QLabel(f"Author: {authorName} Mouse Coords")
    self.imageLbl = QtWidgets.QLabel(f"Image: None")

    self.pxColor = QtWidgets.QLabel("Pixel Color")

    self.mainImg.sigMousePosChanged.connect(lambda pos, pxColor: self.setInfo(pos, pxColor))
    # self.focusedImg.sigMousePosChanged.connect(lambda info: setInfo(info))
    self.statBar.show()
    self.statBar.addWidget(self.imageLbl)
    self.statBar.addWidget(self.mouseCoords)
    self.statBar.addWidget(self.pxColor)


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
    outFname = popupFilePicker(self, 'Select Table Config File', 'All Files (*.*);; Config Files (*.yml)')
    if outFname is not None:
      FR_SINGLETON.tableData.loadCfg(outFname)
      self.resetTblFields()

  def _buildMenu(self):
    # -----
    # MENU BAR
    # -----
    # Top Level
    self.menuFile = QtWidgets.QMenu('&File', self.menuBar_)
    self.menuEdit = QtWidgets.QMenu('&Edit', self.menuBar_)
    self.menuHelp = QtWidgets.QMenu('&Help', self.menuBar_)

    existingActs = self.menuBar_.actions()
    if len(existingActs) == 0:
      leftmost = None
    else:
      leftmost = existingActs[0]

    self.menuBar_.insertMenu(leftmost, self.menuFile)
    self.menuBar_.insertMenu(leftmost, self.menuEdit)
    self.menuBar_.addMenu(self.menuHelp)

    # File / Image
    self.openImgAct = create_addMenuAct(self, self.menuFile, '&Open Image')

    # File / Annotation
    self.openAnnsAct = create_addMenuAct(self, self.menuFile, 'Open &Annotations')
    self.exportAnnsAct = create_addMenuAct(self, self.menuFile, 'E&xport Annotations')

    # File / layout
    self.menuLayout = create_addMenuAct(self, self.menuFile, '&Layout', True)
    self.saveLayoutAct = create_addMenuAct(self, self.menuLayout, 'Save Layout')
    self.menuLayout.addSeparator()

    # File / autosave
    self.menuAutosave = create_addMenuAct(self, self.menuFile, '&Autosave...', True)
    self.startAutosaveAct = create_addMenuAct(self, self.menuAutosave, 'Star&t Autosave')
    self.stopAutosaveAct = create_addMenuAct(self, self.menuAutosave, 'Sto&p Autosave')


    # Edit
    self.undoAct = create_addMenuAct(self, self.menuEdit, '&Undo')
    self.undoAct.setShortcut('Ctrl+Z')
    self.redoAct = create_addMenuAct(self, self.menuEdit, '&Redo')
    self.redoAct.setShortcut('Ctrl+Y')

    # Help
    self.userGuideAct = create_addMenuAct(self, self.menuHelp, 'Online User Guide')
    self.aboutQtAct = create_addMenuAct(self, self.menuHelp, 'Qt Version Info')

    self.setMenuBar(self.menuBar_)


  def _handleNewPlugin(self, plugin: ParamEditorPlugin):
    super()._handleNewPlugin(plugin)
    dock = plugin.dock
    if dock is None:
      return
    FR_SINGLETON.quickLoader.addDock(dock)
    self._tabbifyEditorDocks([dock])

    if plugin.menu is None and plugin.dock is None:
      # No need to add menu and graphics options
      return

    if type(plugin) in _MENU_PLUGINS:
      parentTb = self.menuBar_
    elif isinstance(plugin, TableFieldPlugin):
      parentTb = self.tblFieldToolbar
    else:
      parentTb = self.generalToolbar

    if plugin.dock is None:
      dummyDock = ParamEditorDockGrouping([], plugin.name)
      pluginMenu = self.createMenuOptForDock(dummyDock, parentToolbarOrMenu=parentTb)
    else:
      pluginMenu = self.createMenuOptForDock(plugin.dock, parentToolbarOrMenu=parentTb)

    if plugin.menu is not None:
      if plugin.dock is not None:
        pluginMenu.addSeparator()
      for action in plugin.menu.actions():
        pluginMenu.addAction(action)

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

  def setMainImg(self, fileName: FilePath = None, imgData: NChanImg = None,
                 clearExistingComps=True):
    ret = super().setMainImg(fileName, imgData, clearExistingComps)
    img = self.srcImgFname
    if img is not None:
      img = img.name
    self.imageLbl.setText(f'Image: {img}')

    return ret

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

  def exportAnnotations_gui(self):
    """Saves the component table to a file"""
    fileFilters = self.compIo.handledIoTypes_fileFilter(**{'*': 'All Files'})
    outFname = popupFilePicker(self, 'Select Save File', fileFilters)
    if outFname is not None:
      super().exportAnnotations(outFname)

  def openAnnotation_gui(self):
    # TODO: See note about exporting comps. Delegate the filepicker activity to importer
    fileFilter = self.compIo.handledIoTypes_fileFilter()
    fname = popupFilePicker(self, 'Select Load File', fileFilter)
    if fname is None:
      return
    self.openAnnotations(fname)

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

  def _tabbifyEditorDocks(self, docks):
    # Define out here to retain scope
    def fixDWFactory(_dock):
      # Necessary since defining func in loop will cause problems otherwise
      def doFix(tabIdx):
        self._fixDockWidth(_dock, tabIdx)
      return doFix

    dock = None
    for dock in [FR_SINGLETON.docks[0]] + docks:
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

    editAct = QtWidgets.QAction('Open ' + overrideName, self)
    if editor.saveDir is None:
      # No save options are possible, just use an action instead of dropdown menu
      newMenu = editAct
    else:
      newMenu = QtWidgets.QMenu(overrideName, self)
      newMenu.addAction(editAct)
      newMenu.addSeparator()
      populateFunc = partial(self.populateParamEditorMenuOpts, editor, newMenu, loadFunc)
      editor.sigParamStateCreated.connect(populateFunc)
      # Initialize default menus
      populateFunc()
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
                           parentToolbarOrMenu=None):
    if parentBtn is None:
      parentBtn = QtWidgets.QPushButton()
    if isinstance(dockEditor, ParamEditor):
      parentBtn.setText(dockEditor.name)
      menu = self._createMenuOptForEditor(dockEditor, loadFunc)
      if isinstance(menu, QtWidgets.QMenu):
        parentBtn.setMenu(menu)
      else:
        menu: QtWidgets.QAction
        parentBtn.clicked.connect(menu.triggered.emit)
    else:
      # FRParamEditorDockGrouping
      parentBtn.setText(dockEditor.name)
      menu = QtWidgets.QMenu(dockEditor.name, self)
      parentBtn.setMenu(menu)
      # newMenu = create_addMenuAct(self, parentBtn, dockEditor.name, True)
      for editor in dockEditor.editors:
        # "Main Image Settings" -> "Settings"
        tabName = dockEditor.getTabName(editor)
        nameWithoutBase = tabName
        menuOrAct = self._createMenuOptForEditor(editor, loadFunc, overrideName=nameWithoutBase)
        try:
          menu.addMenu(menuOrAct)
        except TypeError: # Action instead
          menu.addAction(menuOrAct)
    if parentToolbarOrMenu is not None:
      if isinstance(parentToolbarOrMenu, QtWidgets.QToolBar):
        parentToolbarOrMenu.addWidget(parentBtn)
      else:
        # False positive
        # noinspection PyTypeChecker
        parentToolbarOrMenu.addMenu(menu)
    return menu

  def _populateLoadLayoutOptions(self):
    layoutGlob = LAYOUTS_DIR.glob('*.dockstate')
    addDirItemsToMenu(self.menuLayout, layoutGlob, self.loadLayout)

  def setInfo(self, xyPos: XYVertices, pxColor: np.ndarray):
    if pxColor is None: return
    authorName = FR_SINGLETON.tableData.annAuthor

    self.mouseCoords.setText(f'Author: {authorName}'
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