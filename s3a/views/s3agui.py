import warnings
from collections import defaultdict
from pathlib import Path
from typing import Union, Dict, List

import numpy as np
import pyqtgraph as pg
import qdarkstyle
from pandas import DataFrame as df
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
from utilitys import ParamEditor, ParamEditorPlugin, RunOpts, PrjParam, fns, \
  ParamEditorDockGrouping

from s3a.generalutils import hierarchicalUpdate
from s3a.constants import LAYOUTS_DIR, REQD_TBL_FIELDS, ICON_DIR
from s3a.constants import PRJ_ENUMS
from s3a.models.s3abase import S3ABase
from s3a.parameditors import PRJ_SINGLETON
from s3a.plugins.file import FilePlugin
from s3a.plugins.misc import RandomToolsPlugin, MainImagePlugin, CompTablePlugin
from s3a.structures import XYVertices, FilePath, NChanImg

__all__ = ['S3A']

_MENU_PLUGINS = [RandomToolsPlugin]

class S3A(S3ABase):
  sigLayoutSaved = QtCore.Signal()
  S3A_INST = None

  __groupingName__ = 'Main Window'

  @classmethod
  def __initEditorParams__(cls):
    super().__initEditorParams__()
    cls.toolsEditor = ParamEditor.buildClsToolsEditor(cls, 'Main Window')

  def __init__(self, parent=None, guiMode=True, loadLastState=True, **startupSettings):
    # Wait to import quick loader profiles until after self initialization so
    # customized loading functions also get called
    super().__init__(parent, **startupSettings)
    self.setWindowIcon(QtGui.QIcon(str(ICON_DIR/'s3alogo.svg')))
    if guiMode:
      warnings.simplefilter('error', UserWarning)
      fns.makeExceptionsShowDialogs(self)
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
      self.restoreState(fns.attemptFileLoad(layoutName))

    def saveRecentLayout(_folderName: Path):
      outFile = _folderName/'savedLayout.dockstate'
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
    PRJ_SINGLETON.colorScheme.registerFunc(self.updateTheme, runOpts=RunOpts.ON_CHANGED, nest=False)
    # EDIT
    self.saveAllEditorDefaults()

  def _buildGui(self):
    self.setDockOptions(self.ForceTabbedDocks)
    self.setTabPosition(QtCore.Qt.AllDockWidgetAreas, QtWidgets.QTabWidget.North)
    centralWidget = QtWidgets.QWidget()
    self.setCentralWidget(centralWidget)
    layout = QtWidgets.QVBoxLayout(centralWidget)

    self.toolbarWidgets: Dict[PrjParam, List[QtWidgets.QAction]] = defaultdict(list)
    layout.addWidget(self.mainImg)

    self.tblFieldToolbar.setObjectName('Table Field Plugins')
    self.addToolBar(self.tblFieldToolbar)
    self.generalToolbar.setObjectName('General')
    self.addToolBar(self.generalToolbar)

    _plugins = [PRJ_SINGLETON.clsToPluginMapping[c] for c in [MainImagePlugin, CompTablePlugin]]
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

    self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, tableDock)

    # STATUS BAR
    self.setStatusBar(self.statBar)

    self.mouseCoordsLbl = QtWidgets.QLabel()
    self.pxColorLbl = QtWidgets.QLabel()

    self.imageLbl = QtWidgets.QLabel(f"Image: None")

    self.statBar.show()
    self.statBar.addWidget(self.imageLbl)

    self.statBar.addWidget(self.mouseCoordsLbl)
    self.mainImg.mouseCoordsLbl = self.mouseCoordsLbl

    self.statBar.addWidget(self.pxColorLbl)
    self.mainImg.pxColorLbl = self.pxColorLbl

  def saveLayout(self, layoutName: Union[str, Path]=None, allowOverwriteDefault=False):
    dockStates = self.saveState().data()
    if Path(layoutName).is_absolute():
      savePathPlusStem = layoutName
    else:
      savePathPlusStem = LAYOUTS_DIR/layoutName
    saveFile = savePathPlusStem.with_suffix(f'.dockstate')
    fns.saveToFile(dockStates, saveFile,
                   allowOverwriteDefault=allowOverwriteDefault)
    self.sigLayoutSaved.emit()

  def changeFocusedComp(self, newComps: df=None, forceKeepLastChange=False):
    ret = super().changeFocusedComp(newComps, forceKeepLastChange)
    self.curCompIdLbl.setText(f'Component ID: {self.mainImg.compSer[REQD_TBL_FIELDS.INST_ID]}')
    return ret

  def resetTblFields_gui(self):
    outFname = fns.popupFilePicker(None, 'Select Table Config File', 'All Files (*.*);; Config Files (*.yml)')
    if outFname is not None:
      PRJ_SINGLETON.tableData.loadCfg(outFname)

  def _buildMenu(self):
    # TODO: Find a better way of fixing up menu order
    menus = self.menuBar_.actions()
    menuFile = [a for a in menus if a.text() == FilePlugin.name][0]
    self.menuBar_.insertAction(menus[0], menuFile)

  def _handleNewPlugin(self, plugin: ParamEditorPlugin):
    super()._handleNewPlugin(plugin)
    dock = plugin.dock
    if dock is None:
      return
    PRJ_SINGLETON.quickLoader.addDock(dock)
    self.addTabbedDock(QtCore.Qt.RightDockWidgetArea, dock)

    if plugin.menu is None:
      # No need to add menu and graphics options
      return

    parentTb = plugin.parentMenu
    if parentTb is not None:
      self.createMenuOptForPlugin(plugin, parentToolbarOrMenu=parentTb)

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
    fname = fns.popupFilePicker(None, 'Select Main Image', fileFilter)
    if fname is not None:
      with pg.BusyCursor():
        self.setMainImg(fname)

  def exportAnnotations_gui(self):
    """Saves the component table to a file"""
    fileFilters = self.compIo.handledIoTypes_fileFilter(**{'*': 'All Files'})
    outFname = fns.popupFilePicker(None, 'Select Save File', fileFilters, existing=False)
    if outFname is not None:
      super().exportCurAnnotation(outFname)

  def openAnnotation_gui(self):
    # TODO: See note about exporting comps. Delegate the filepicker activity to importer
    fileFilter = self.compIo.handledIoTypes_fileFilter()
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
      msg.setText('Component table has unsaved changes.\nAre you sure you want to exit?')
      msg.setDefaultButton(msg.Ok)
      msg.setStandardButtons(msg.Discard|msg.Cancel|msg.Ok)
      code = msg.exec_()
      if code == msg.Discard:
        forceClose = True
      elif code == msg.Cancel:
        shouldExit = False
    if shouldExit:
      # Clean up all editor windows, which could potentially be left open
      ev.accept()
      PRJ_SINGLETON.close()
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

  def createMenuOptForPlugin(self, plugin: ParamEditorPlugin, parentToolbarOrMenu=None):
    if isinstance(parentToolbarOrMenu, QtWidgets.QToolBar):
      btn = QtWidgets.QToolButton()
      btn.setText(plugin.name)
      parentToolbarOrMenu.addWidget(btn)
      parentToolbarOrMenu = btn
      btn.addMenu = btn.setMenu
      btn.addAction = lambda act: act.triggered.connect(btn.click)
      btn.setPopupMode(btn.InstantPopup)
    parentToolbarOrMenu.addMenu(plugin.menu)
    oldShow = plugin.dock.showEvent
    def show_fixDockWidth(ev):
      oldShow(ev)
      dock = plugin.dock
      dock.raise_()
      dock.activateWindow()
      if dock.width() < dock.biggestMinWidth + 100:
        self.resizeDocks([dock], [dock.biggestMinWidth + 100], QtCore.Qt.Horizontal)
    plugin.dock.showEvent = show_fixDockWidth

  def _populateLoadLayoutOptions(self):
    self.layoutEditor.addDirItemsToMenu(self.menuLayout)

  def updateTheme(self, useDarkTheme=False):
    style = ''
    if useDarkTheme:
      style = qdarkstyle.load_stylesheet()
    self.setStyleSheet(style)

  def add_focusComps(self, newComps: df, addType=PRJ_ENUMS.COMP_ADD_AS_NEW):
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