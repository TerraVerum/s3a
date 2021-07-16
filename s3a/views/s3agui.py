import logging
from collections import defaultdict
from pathlib import Path
from typing import Union, Dict, List, Sequence

import pyqtgraph as pg
import qdarkstyle
from pandas import DataFrame as df
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui

from ..constants import LAYOUTS_DIR, REQD_TBL_FIELDS, ICON_DIR, PRJ_ENUMS, PRJ_CONSTS
from ..logger import getAppLogger
from s3a.generalutils import hierarchicalUpdate
from s3a.models.s3abase import S3ABase
from s3a.plugins.misc import RandomToolsPlugin, MainImagePlugin, CompTablePlugin
from s3a.shared import SharedAppSettings
from s3a.structures import FilePath, NChanImg
from utilitys import ParamEditor, ParamEditorPlugin, RunOpts, PrjParam, fns, widgets

__all__ = ['S3A']

_MENU_PLUGINS = [RandomToolsPlugin]

class S3A(S3ABase):
  sigLayoutSaved = QtCore.Signal()

  __groupingName__ = 'Main Window'

  def __initEditorParams__(self, shared: SharedAppSettings):
    super().__initEditorParams__(shared)
    self.toolsEditor = ParamEditor.buildClsToolsEditor(type(self), 'Main Window')
    shared.colorScheme.registerFunc(self.updateTheme, runOpts=RunOpts.ON_CHANGED, nest=False)


  def __init__(self, parent=None, log: Union[str, Sequence[str]]=PRJ_ENUMS.LOG_TERM,
               loadLastState=True, **startupSettings):
    # Wait to import quick loader profiles until after self initialization so
    # customized loading functions also get called
    super().__init__(parent, **startupSettings)
    self.setWindowIcon(QtGui.QIcon(str(ICON_DIR/'s3alogo.svg')))
    logger = getAppLogger()
    if PRJ_ENUMS.LOG_GUI in log:
      logger.registerExceptions()
      logger.registerWarnings()
      logger.addHandler(widgets.FadeNotifyHandler(PRJ_ENUMS.LOG_LVL_ATTN, self, maxLevel=PRJ_ENUMS.LOG_LVL_ATTN))
      logger.addHandler(widgets.StatusBarHandler(logging.INFO, self, maxLevel=logging.INFO))
      # This logger isn't supposed to propagate, since everything is handled in the terminal on accepted events
      # unless 'terminal' is also specified
      if PRJ_ENUMS.LOG_TERM not in log:
        logger.propagate = False
    self.APP_TITLE = 'FICS Semi-Supervised Semantic Annotator'
    self.CUR_COMP_LBL = 'Current Component ID:'
    self.setWindowTitle(self.APP_TITLE)
    self.setWindowIconText(self.APP_TITLE)

    self.curCompIdLbl = QtWidgets.QLabel(self.CUR_COMP_LBL)

    # -----
    # LAOYUT MANAGER
    # -----
    # Dummy editor for layout options since it doesn't really have editable settings
    # Maybe later this can be elevated to have more options
    self.layoutEditor = ParamEditor(self, None, LAYOUTS_DIR, 'dockstate', 'Layout')
    def loadLayout(layoutName: Union[str, Path]):
      layoutName = Path(layoutName)
      if not layoutName.is_absolute():
        layoutName = LAYOUTS_DIR/f'{layoutName}.dockstate'
      state = fns.attemptFileLoad(layoutName)
      self.restoreState(state['docks'])

    def saveRecentLayout(_folderName: Path):
      outFile = _folderName/'layout.dockstate'
      self.saveLayout(outFile)
      return str(outFile)

    self.layoutEditor.loadParamValues = loadLayout
    self.layoutEditor.saveParamValues = saveRecentLayout
    self.appStateEditor.addImportExportOpts('layout', loadLayout, saveRecentLayout)

    self._buildGui()
    self._buildMenu()
    self._hookupSignals()

    # Load in startup settings
    stateDict = None if loadLastState else {}
    hierarchicalUpdate(self.appStateEditor.startupSettings, startupSettings)
    self.appStateEditor.loadParamValues(stateDict=stateDict)

  def _hookupSignals(self):
    # EDIT
    self.saveAllEditorDefaults()

  def _buildMenu(self):
    # Nothing to do for now
    pass

  def _buildGui(self):
    self.setDockOptions(self.ForceTabbedDocks)
    self.setTabPosition(QtCore.Qt.AllDockWidgetAreas, QtWidgets.QTabWidget.North)
    centralWidget = QtWidgets.QWidget()
    self.setCentralWidget(centralWidget)
    layout = QtWidgets.QVBoxLayout(centralWidget)

    self.toolbarWidgets: Dict[PrjParam, List[QtWidgets.QAction]] = defaultdict(list)
    layout.addWidget(self.mainImg)

    self.tblFieldToolbar.setObjectName('Table Field Plugins')
    # self.addToolBar(self.tblFieldToolbar)
    self.tblFieldToolbar.hide()
    self.generalToolbar.setObjectName('General')
    self.addToolBar(self.generalToolbar)

    _plugins = [self.clsToPluginMapping[c] for c in [MainImagePlugin, CompTablePlugin]]
    parents = [self.mainImg, self.compTbl]
    for plugin, parent in zip(_plugins, reversed(parents)):
      plugin.toolsEditor.actionsMenuFromProcs(plugin.name, nest=True, parent=parent, outerMenu=parent.menu)


    tableDock = QtWidgets.QDockWidget('Component Table Window', self)
    tableDock.setFeatures(tableDock.DockWidgetMovable|tableDock.DockWidgetFloatable)

    tableDock.setObjectName('Component Table Dock')
    tableContents = QtWidgets.QWidget(tableDock)
    tableLayout = QtWidgets.QVBoxLayout(tableContents)
    tableLayout.addWidget(self.compTbl)
    tableDock.setWidget(tableContents)

    self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, tableDock)

    # STATUS BAR
    self.setStatusBar(self.statBar)

    self.mouseCoordsLbl = QtWidgets.QLabel()
    self.pxColorLbl = QtWidgets.QLabel()

    self.imageLbl = QtWidgets.QLabel(f"Image: None")

    self.statBar.show()
    self.statBar.addPermanentWidget(self.imageLbl)

    self.statBar.addPermanentWidget(self.mouseCoordsLbl)
    self.mainImg.mouseCoordsLbl = self.mouseCoordsLbl

    self.statBar.addPermanentWidget(self.pxColorLbl)
    self.mainImg.pxColorLbl = self.pxColorLbl

  def saveLayout(self, layoutName: Union[str, Path]=None, allowOverwriteDefault=False):
    dockStates = self.saveState().data()
    if Path(layoutName).is_absolute():
      savePathPlusStem = layoutName
    else:
      savePathPlusStem = LAYOUTS_DIR/layoutName
    saveFile = savePathPlusStem.with_suffix(f'.dockstate')
    fns.saveToFile({'docks': dockStates}, saveFile,
                   allowOverwriteDefault=allowOverwriteDefault)
    self.sigLayoutSaved.emit()

  def changeFocusedComp(self, compIds: Union[int, Sequence[int]]=None):
    ret = super().changeFocusedComp(compIds)
    self.curCompIdLbl.setText(f'Component ID: {self.mainImg.compSer[REQD_TBL_FIELDS.INST_ID]}')
    return ret

  def resetTblFields_gui(self):
    outFname = fns.popupFilePicker(None, 'Select Table Config File', 'All Files (*.*);; Config Files (*.yml)')
    if outFname is not None:
      self.sharedAttrs.tableData.loadCfg(outFname)

  def _addPluginObj(self, plugin: ParamEditorPlugin, **kwargs):
    plugin = super()._addPluginObj(plugin, **kwargs)
    if not plugin:
      return
    dock = plugin.dock
    if dock is None:
      return
    self.sharedAttrs.quickLoader.addDock(dock)
    self.addTabbedDock(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)

    if plugin.menu is None:
      # No need to add menu and graphics options
      return plugin

    parentTb = plugin.parentMenu
    if parentTb is not None:
      plugin.addToWindow(self, parentToolbarOrMenu=parentTb)
    return plugin

  def setMainImg(self, fileName: FilePath = None, imgData: NChanImg = None,
                 clearExistingComps=True):
    gen = super().setMainImg(fileName, imgData, clearExistingComps)
    ret = fns.gracefulNext(gen)
    img = self.srcImgFname
    if img is not None:
      img = img.name
    self.imageLbl.setText(f'Image: {img}')

    yield ret
    yield fns.gracefulNext(gen)

  def setMainImg_gui(self):
    fileFilter = "Image Files (*.png *.tif *.jpg *.jpeg *.bmp *.jfif);;All files(*.*)"
    fname = fns.popupFilePicker(None, 'Select Main Image', fileFilter)
    if fname is not None:
      with pg.BusyCursor():
        self.setMainImg(fname)

  def exportAnnotations_gui(self):
    """Saves the component table to a file"""
    fileFilters = self.compIo.ioFileFilter(**{'*': 'All Files'})
    outFname = fns.popupFilePicker(None, 'Select Save File', fileFilters, existing=False)
    if outFname is not None:
      super().exportCurAnnotation(outFname)

  def openAnnotation_gui(self):
    # TODO: See note about exporting comps. Delegate the filepicker activity to importer
    fileFilter = self.compIo.ioFileFilter(which=PRJ_ENUMS.IO_IMPORT)
    fname = fns.popupFilePicker(None, 'Select Load File', fileFilter)
    if fname is None:
      return
    self.openAnnotations(fname)

  def saveLayout_gui(self):
    outName = fns.dialogGetSaveFileName(self, 'Layout Name')
    if outName is None or outName == '':
      return
    self.saveLayout(outName)

  # ---------------
  # BUTTON CALLBACKS
  # ---------------
  def closeEvent(self, ev: QtGui.QCloseEvent):
    # Confirm all components have been saved
    shouldExit = True
    forceClose = False
    if self.hasUnsavedChanges:
      ev.ignore()
      forceClose = False
      msg = QtWidgets.QMessageBox()
      msg.setWindowTitle('Confirm Exit')
      msg.setText('Component table has unsaved changes.\nYou can choose to save and exit or discard changes')
      msg.setDefaultButton(msg.Save)
      msg.setStandardButtons(msg.Discard|msg.Cancel|msg.Save)
      code = msg.exec_()
      if code == msg.Discard:
        forceClose = True
      elif code == msg.Cancel:
        shouldExit = False
    if shouldExit:
      # Clean up all editor windows, which could potentially be left open
      ev.accept()
      fns.restoreExceptionBehavior()
      if not forceClose:
        self.appStateEditor.saveParamValues()

  def forceClose(self):
    """
    Allows the app to close even if it has unsaved changes. Useful for closing
    within a script
    """
    self.hasUnsavedChanges = False
    self.close()

  def _populateLoadLayoutOptions(self):
    self.layoutEditor.addDirItemsToMenu(self.menuLayout)

  def updateTheme(self, useDarkTheme=False):
    style = ''
    if useDarkTheme:
      style = qdarkstyle.load_stylesheet()
    self.setStyleSheet(style)

  def add_focusComps(self, newComps: df, addType=PRJ_ENUMS.COMP_ADD_AS_NEW):
    changeDict = super().add_focusComps(newComps, addType=addType)
    keepIds = changeDict['ids']
    keepIds = keepIds[keepIds >= 0]
    selection = self.compDisplay.selectRowsById(keepIds)
    if self.isVisible() and self.compTbl.props[PRJ_CONSTS.PROP_SHOW_TBL_ON_COMP_CREATE]:
      # For some reason sometimes the actual table selection doesn't propagate in time, so
      # directly forward the selection here
      self.compTbl.setSelectedCellsAs_gui(selection)
    return changeDict

if __name__ == '__main__':
  import sys
  app = QtWidgets.QApplication([])
  win = S3A()
  win.showMaximized()
  sys.exit(app.exec_())