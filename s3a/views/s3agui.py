import warnings
from collections import defaultdict
from pathlib import Path
from typing import Union, Dict, List

import numpy as np
import pyqtgraph as pg
import qdarkstyle
from pandas import DataFrame as df
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui

from s3a import RunOpts, PrjParam
from s3a.constants import PRJ_ENUMS
from s3a.constants import LAYOUTS_DIR, PRJ_CONSTS as CNST, REQD_TBL_FIELDS
from s3a.generalutils import attemptFileLoad
from s3a.graphicsutils import makeExceptionsShowDialogs, popupFilePicker, \
  disableAppDuringFunc, saveToFile, dialogGetSaveFileName, addDirItemsToMenu, \
  restoreExceptionBehavior, menuFromEditorActions
from s3a.models.s3abase import S3ABase
from s3a.parameditors import ParamEditor, FR_SINGLETON
from s3a.plugins.base import ParamEditorPlugin
from s3a.plugins.file import FilePlugin
from s3a.plugins.misc import RandomToolsPlugin, MainImagePlugin, CompTablePlugin
from s3a.structures import S3AWarning, XYVertices, FilePath, NChanImg

__all__ = ['S3A']

_MENU_PLUGINS = [RandomToolsPlugin]

class S3A(S3ABase):
  sigLayoutSaved = QtCore.Signal()
  S3A_INST = None

  __groupingName__ = 'Application'

  @classmethod
  def __initEditorParams__(cls):
    super().__initEditorParams__()
    cls.toolsEditor = ParamEditor.buildClsToolsEditor(cls, 'Main Window')

  def __init__(self, parent=None, guiMode=True, loadLastState=False,
               **startupSettings):
    # Wait to import quick loader profiles until after self initialization so
    # customized loading functions also get called
    superLoaderArgs = {'author': startupSettings.pop('author', None)}
    super().__init__(parent, **superLoaderArgs)
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

    # Load layout options
    self.saveLayout('Default', allowOverwriteDefault=True)
    stateDict = None if loadLastState else {}
    with pg.BusyCursor():
      self.appStateEditor.loadParamState(stateDict=stateDict, overrideDict=startupSettings)

  def _hookupSignals(self):
    FR_SINGLETON.colorScheme.registerFunc(self.updateTheme, runOpts=RunOpts.ON_CHANGED, nest=False)
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

    _plugins = [FR_SINGLETON.clsToPluginMapping[c] for c in [MainImagePlugin, CompTablePlugin]]
    parents = [self.mainImg, self.compTbl]
    for plugin, parent in zip(_plugins, reversed(parents)):
      parent.menu.addMenu(menuFromEditorActions([plugin.toolsEditor], plugin.name, menuParent=parent, nest=False))


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

  def changeFocusedComp(self, newComps: df=None, forceKeepLastChange=False):
    ret = super().changeFocusedComp(newComps, forceKeepLastChange)
    self.curCompIdLbl.setText(f'Component ID: {self.focusedImg.compSer[REQD_TBL_FIELDS.INST_ID]}')
    return ret

  def resetTblFields_gui(self):
    outFname = popupFilePicker(None, 'Select Table Config File', 'All Files (*.*);; Config Files (*.yml)')
    if outFname is not None:
      FR_SINGLETON.tableData.loadCfg(outFname)

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
    FR_SINGLETON.quickLoader.addDock(dock)
    self.addTabbedDock(QtCore.Qt.RightDockWidgetArea, dock)

    if plugin.menu is None:
      # No need to add menu and graphics options
      return

    parentTb = plugin.parentMenu

    self.createMenuOptForPlugin(plugin, parentToolbarOrMenu=parentTb)

  def _maybeLoadLastState_gui(self, loadLastState: bool=None,
                              startupSettings:dict=None):
    """
    Helper function to determine whether the last application state should be loaded,
    and loads the last state if desired.
    :param loadLastState: If *None*, the user will be prompted via dialog for whether
      to load the last application state. Otherwise, its boolean value is used.
    :param startupSettings: Additional dict arguments which if supplied will override
      the default options where applicable.
    """
    if loadLastState is None:
      loadLastState = QtWidgets.QMessageBox.question(
        self, 'Load Previous State', 'Do you want to load all previous app'
                                     ' settings (image, annotations, algorithms, etc.)?',
        QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No) == QtWidgets.QMessageBox.Yes
    if loadLastState:
      self.loadLastState_gui(startupSettings)

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
    fname = popupFilePicker(None, 'Select Main Image', fileFilter)
    if fname is not None:
      with pg.BusyCursor():
        self.setMainImg(fname)

  def exportAnnotations_gui(self):
    """Saves the component table to a file"""
    fileFilters = self.compIo.handledIoTypes_fileFilter(**{'*': 'All Files'})
    outFname = popupFilePicker(None, 'Select Save File', fileFilters, existing=False)
    if outFname is not None:
      super().exportCurAnnotation(outFname)

  def openAnnotation_gui(self):
    # TODO: See note about exporting comps. Delegate the filepicker activity to importer
    fileFilter = self.compIo.handledIoTypes_fileFilter()
    fname = popupFilePicker(None, 'Select Load File', fileFilter)
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

  # Stolen and adapted for python from https://stackoverflow.com/a/42910109/9463643
  # noinspection PyTypeChecker
  def addTabbedDock(self, area: QtCore.Qt.DockWidgetArea, dockwidget: QtWidgets.QDockWidget):
    curAreaWidgets = [d for d in self.findChildren(QtWidgets.QDockWidget)
                      if self.dockWidgetArea(d) == area]
    try:
      self.tabifyDockWidget(curAreaWidgets[-1], dockwidget)
    except IndexError:
      # First dock in area
      self.addDockWidget(area, dockwidget)

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
      plugin.dock.raise_()
      plugin.dock.activateWindow()
      if plugin.dock.width() < plugin.dock.biggestMinWidth + 100:
        self.resizeDocks([plugin.dock], [plugin.dock.biggestMinWidth + 100], QtCore.Qt.Horizontal)
    plugin.dock.showEvent = show_fixDockWidth

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

  def add_focusComps(self, newComps: df, addType=PRJ_ENUMS.COMP_ADD_AS_NEW):
    ret = super().add_focusComps(newComps, addType=addType)
    selection = self.compDisplay.selectRowsById(newComps[REQD_TBL_FIELDS.INST_ID])
    if self.isVisible() and self.compTbl.showOnCreate:
      # For some reason sometimes the actual table selection doesn't propagate in time, so
      # directly forward the selection here
      self.compTbl.setSelectedCellsAs_gui(selection)
    return ret

  def savePlotScreenshot(self, outFname:FilePath=None):
    """
    Saves main image and the plot of components to a file
    :param outFname:
      helpText: Where to save the image
      pType: filepicker
      existing: False
    """
    if outFname is None:
      return
    outFname = Path(outFname)
    pixmap = self.mainImg.imgItem.getPixmap()
    painter = QtGui.QPainter(pixmap)
    self.compDisplay.regionPlot.paint(painter)
    painter.end()
    self.mainImg.render()

if __name__ == '__main__':
  import sys
  app = QtWidgets.QApplication([])
  win = S3A()
  win.showMaximized()
  sys.exit(app.exec_())