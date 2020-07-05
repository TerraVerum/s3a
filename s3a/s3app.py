# -*- coding: utf-8 -*-

import sys
import warnings
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Any, Union, Optional, Tuple
from warnings import warn

import pandas as pd
import numpy as np
import qdarkstyle
from pandas import DataFrame as df
from pyqtgraph import BusyCursor
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui

from .frgraphics.parameditors import FR_SINGLETON
from .frgraphics.annotator_ui import FRAnnotatorUI
from .frgraphics.graphicsutils import (dialogGetSaveFileName, addDirItemsToMenu,
                                       attemptFileLoad, popupFilePicker,
                                       disableAppDuringFunc, makeExceptionsShowDialogs,
                                       autosaveOptsDialog, create_addMenuAct,
                                       raiseErrorLater)
from .frgraphics.graphicsutils import saveToFile
from .frgraphics.imageareas import FREditableImg
from .frgraphics.parameditors import FRParamEditor, FRParamEditorDockGrouping
from .generalutils import resolveAuthorName
from .projectvars.constants import FR_CONSTS
from .projectvars.constants import LAYOUTS_DIR, REQD_TBL_FIELDS
from .projectvars.enums import FR_ENUMS, _FREnums
from .structures import FRAppIOError, NChanImg, FilePath, FRVertices, FRAlgProcessorError, \
  FRS3AWarning
from .tablemodel import FRComponentIO
from .tablemodel import FRComponentMgr
from .tableviewproxy import FRCompDisplayFilter, FRCompSortFilter

Signal = QtCore.Signal

@FR_SINGLETON.registerGroup(FR_CONSTS.CLS_ANNOTATOR)
class S3A(FRAnnotatorUI):
  """
  Top-level widget for producing component bounding boxes from an input image.
  """
  # Alerts GUI that a layout (either new or overwriting old) was saved
  sigLayoutSaved = Signal()

  @classmethod
  def __initEditorParams__(cls):
    cls.estBoundsOnStart, cls.undoBuffSz = FR_SINGLETON.generalProps.registerProps(cls,
        [FR_CONSTS.PROP_EST_BOUNDS_ON_START, FR_CONSTS.PROP_UNDO_BUF_SZ])
    cls.useDarkTheme = FR_SINGLETON.scheme.registerProp(cls, FR_CONSTS.SCHEME_USE_DARK_THEME)

  def __init__(self, exceptionsAsDialogs=True, **quickLoaderArgs):
    super().__init__()
    if exceptionsAsDialogs:
      warnings.simplefilter('error', FRS3AWarning)
      makeExceptionsShowDialogs(self)

    FR_SINGLETON.sigDocksAdded.connect(lambda newDocks: self.addEditorDocks(newDocks))
    self.addEditorDocks()
    # ---------------
    # DATA ATTRIBUTES
    # ---------------
    self.hasUnsavedChanges = False
    self.srcImgFname = None
    self.autosaveTimer: Optional[QtCore.QTimer] = None

    self.statBar = QtWidgets.QStatusBar(self)
    self.setStatusBar(self.statBar)
    authorName = resolveAuthorName(quickLoaderArgs.pop('Author', None))
    if authorName is None:
      sys.exit('No author name provided and no default author exists. Exiting.\n'
               'To start without error, provide an author name explicitly, e.g.\n'
               '"python -m s3a --author=<Author Name>"')
    FR_SINGLETON.tableData.annAuthor = authorName

    self.mouseCoords = QtWidgets.QLabel(f"Author: {authorName} Mouse Coords")

    self.pxColor = QtWidgets.QLabel("Pixel Color")

    self.mainImg.sigMousePosChanged.connect(lambda info: self.setInfo(info))
    # self.focusedImg.sigMousePosChanged.connect(lambda info: setInfo(info))
    self.statBar.show()
    self.statBar.addWidget(self.mouseCoords)
    self.statBar.addWidget(self.pxColor)

    # Flesh out pg components
    # ---------------
    # MAIN IMAGE
    # ---------------
    self.mainImg.sigComponentsCreated.connect(self.add_focusComp)

    def handleUpdate(comps):
      self.compMgr.addComps(comps, FR_ENUMS.COMP_ADD_AS_MERGE)
      self.changeFocusedComp(comps)
    self.mainImg.sigComponentsUpdated.connect(handleUpdate)

    self.mainImg.sigComponentsRemoved.connect(
      lambda compIds: self.compMgr.rmComps(compIds)
    )

    # ---------------
    # COMPONENT MANAGER
    # ---------------
    self.compExporter = FRComponentIO()
    self.compMgr = FRComponentMgr()
    self.compMgr.sigCompsChanged.connect(self._recordCompChange)

    # Allow filtering/sorting
    self.sortFilterProxy = FRCompSortFilter(self.compMgr, self)

    self.compTbl.setModel(self.sortFilterProxy)

    # ---------------
    # COMPONENT DISPLAY FILTER
    # ---------------
    self.compDisplay = FRCompDisplayFilter(self.compMgr, self.mainImg, self.compTbl,
                                           self)

    self.mainImg.imgItem.sigImageChanged.connect(self.clearBoundaries)
    self.compDisplay.sigCompsSelected.connect(lambda newComps: self.changeFocusedComp(newComps))

    # ---------------
    # UI ELEMENT SIGNALS
    # ---------------
    # Buttons
    self.openImgAct.triggered.connect(lambda: self.openImgActionTriggered())
    self.clearRegionBtn.clicked.connect(self.clearRegionBtnClicked)
    self.resetRegionBtn.clicked.connect(self.resetRegionBtnClicked)
    self.acceptRegionBtn.clicked.connect(self.acceptRegionBtnClicked)

    FR_SINGLETON.scheme.sigParamStateUpdated.connect(self.updateTheme)
    FR_SINGLETON.generalProps.sigParamStateUpdated.connect(self.updateUndoBuffSz)

    # Menu options
    # FILE
    self.saveLayoutAct.triggered.connect(self.saveLayoutActionTriggered)
    self.sigLayoutSaved.connect(self.populateLoadLayoutOptions)

    self.exportCompListAct.triggered.connect(self.exportCompListActionTriggered)
    self.exportLabelImgAct.triggered.connect(self.exportLabelImgActionTriggered)
    self.loadCompsAct_merge.triggered.connect(lambda: self.loadCompsActionTriggered(FR_ENUMS.COMP_ADD_AS_MERGE))
    self.loadCompsAct_new.triggered.connect(lambda: self.loadCompsActionTriggered(FR_ENUMS.COMP_ADD_AS_NEW))
    self.startAutosaveAct.triggered.connect(self.autosaveAcionTriggered)
    self.stopAutosaveAct.triggered.connect(self.stopAutosave)

    # SETTINGS
    for dock in FR_SINGLETON.docks:
        self.createMenuOptForDock(self.paramTools, dock)
    self.createMenuOptForEditor(self.menuFile, FR_SINGLETON.quickLoader,
                                self.importQuickLoaderProfile)
    if quickLoaderArgs is not None:
      self.importQuickLoaderProfile(quickLoaderArgs)

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

    # Load layout options
    self.saveLayout('Default', allowOverwriteDefault=True)

  # -----------------------------
  # S3A CLASS FUNCTIONS
  # -----------------------------
  # -----
  # Gui stuff
  # -----
  def closeEvent(self, ev: QtGui.QCloseEvent):
    # Confirm all components have been saved
    shouldExit = False
    if self.hasUnsavedChanges:
      ev.ignore()
      if (QtWidgets.QMessageBox.question(self, 'Confirm Exit',
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

  def addEditorDocks(self, docks=None):
    if docks is None:
      docks = FR_SINGLETON.docks
    # Define out here to retain scope
    dock = None
    for dock in docks:
      dock.setParent(self)
      self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
    for nextEditor in docks[:-1]:
      self.tabifyDockWidget(dock, nextEditor)

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

  def createMenuOptForEditor(self, parentMenu: QtWidgets.QMenu, editor: FRParamEditor,
                             loadFunc=None, overrideName=None):
    if overrideName is None:
      overrideName = editor.name
    if editor.hasMenuOption:
      return
    if loadFunc is None:
      loadFunc = partial(self.paramEditorLoadActTriggered, editor)
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

  @staticmethod
  def populateParamEditorMenuOpts(objForMenu: FRParamEditor, winMenu: QtWidgets.QMenu,
                                  triggerFn: Callable):
    addDirItemsToMenu(winMenu,
                      objForMenu.saveDir.glob(f'*.{objForMenu.fileType}'),
                      triggerFn)

  # -----
  # App functionality
  # -----
  def setInfo(self, info: Tuple[Tuple[int, int], np.ndarray]):
    authorName = FR_SINGLETON.tableData.annAuthor
    rowColCoords = info[0]
    color = info[1]
    self.mouseCoords.setText(f'Author: {authorName} | Mouse (x,y): {rowColCoords[1]}, {rowColCoords[0]} | Pixel Color: ')
    self.pxColor.setText(f'{color}')
    if color.dtype == float:
      # Turn to uint
      color = (color*255).astype('uint8')
    # Regardless of the number of image channels, display as RGBA color
    if color.size == 1:
      color = np.array([color[0]]*3 + [255])
    elif color.size == 3:
      color = np.concatenate([color, [255]])
    # Else: assume already RGBA
    # Determine text color based on background color
    if np.mean(color) > 127:
      fontColor = 'black'
    else:
      fontColor = 'white'
    self.pxColor.setStyleSheet(f'background:rgba{tuple(color)}; color:{fontColor}; font-weight: 16px')

  def startAutosave(self, interval_mins: float, autosaveFolder: Path, baseName: str):
    autosaveFolder.mkdir(exist_ok=True, parents=True)
    lastSavedDf = self.compMgr.compDf.copy()
    # Qtimer expects ms, turn mins->s->ms
    self.autosaveTimer = QtCore.QTimer(self)
    # Figure out where to start the counter
    globExpr = lambda: autosaveFolder.glob(f'{baseName}*.csv')
    existingFiles = list(globExpr())
    if len(existingFiles) == 0:
      counter = 0
    else:
      counter = max(map(lambda fname: int(fname.stem.rsplit('_')[1]), existingFiles)) + 1

    def save_incrementCounter():
      nonlocal counter, lastSavedDf
      baseSaveNamePlusFolder = autosaveFolder/f'{baseName}_{counter}.csv'
      counter += 1
      if not np.array_equal(self.compMgr.compDf, lastSavedDf):
        self.exportCompList(baseSaveNamePlusFolder)
        lastSavedDf = self.compMgr.compDf.copy()

    self.autosaveTimer.timeout.connect(save_incrementCounter)
    self.autosaveTimer.start(int(interval_mins*60*1000))

  def stopAutosave(self):
    self.autosaveTimer.stop()

  def autosaveAcionTriggered(self):
    saveDlg = autosaveOptsDialog(self)
    success = saveDlg.exec()
    if success:
      try:
        interval = saveDlg.intervalEdit.value()
        baseName = saveDlg.baseFileNameEdit.text()
        folderName = Path(saveDlg.folderName)
      except AttributeError:
        warn('Some information was not provided -- autosave not started.', FRS3AWarning)
      else:
        self.startAutosave(interval, folderName, baseName)

  def updateTheme(self, _newScheme: Dict[str, Any]):
    style = ''
    if self.useDarkTheme:
      style = qdarkstyle.load_stylesheet()
    self.setStyleSheet(style)
    for opts in self.focusedImg.drawOptsWidget, self.mainImg.drawOptsWidget:
      opts.horizWidth = opts.layout().minimumSize().width()

  def updateUndoBuffSz(self, _genProps: Dict[str, Any]):
    FR_SINGLETON.actionStack.resizeStack(self.undoBuffSz)

  def clearFocusedRegion(self):
    # Reset drawn comp vertices to nothing
    # Only perform action if image currently exists
    if self.focusedImg.image is None:
      return
    self.focusedImg.updateRegionFromVerts(None)

  def resetFocusedRegion(self):
    # Reset drawn comp vertices to nothing
    # Only perform action if image currently exists
    if self.focusedImg.image is None:
      return
    self.focusedImg.updateRegionFromVerts(self.focusedImg.compSer[REQD_TBL_FIELDS.VERTICES])

  @FR_SINGLETON.actionStack.undoable('Accept Focused Region')
  def acceptFocusedRegion(self):
    # If the component was deleted
    focusedId = self.focusedImg.compSer[REQD_TBL_FIELDS.INST_ID]
    if focusedId not in self.compMgr.compDf.index:
      warn('Cannot accept region as this component was deleted.', FRS3AWarning)
      return
    oldSer = self.compMgr.compDf.loc[focusedId].copy()

    self.focusedImg.saveNewVerts()
    modifiedComp = self.focusedImg.compSer
    modified_df = modifiedComp.to_frame().T
    # keepCols = [REQD_TBL_FIELDS.VERTICES, REQD_TBL_FIELDS.INST_ID]
    # xpondingComp = self.compMgr.compDf[[modifiedComp[REQD_TBL_FIELDS.INST_ID]]].copy()
    # xpondingComp.at[0, REQD_TBL_FIELDS.VERTICES] = modifiedComp[REQD_TBL_FIELDS.VERTICES]
    self.compMgr.addComps(modified_df, addtype=FR_ENUMS.COMP_ADD_AS_MERGE)
    self.compDisplay.regionPlot.focusById([modifiedComp[REQD_TBL_FIELDS.INST_ID]])
    yield
    self.focusedImg.saveNewVerts(oldSer[REQD_TBL_FIELDS.VERTICES])
    self.compMgr.addComps(oldSer.to_frame().T, addtype=FR_ENUMS.COMP_ADD_AS_MERGE)


  def estimateBoundaries(self):
    self.mainImg.handleShapeFinished(FRVertices())

  def clearBoundaries(self):
    self.compMgr.rmComps()

  @FR_SINGLETON.actionStack.undoable('Change Main Image')
  def resetMainImg(self, fileName: FilePath=None, imgData: NChanImg=None,
                   clearExistingComps=True):
    """
    * If fileName is None, the main and focused images are blacked out.
    * If only fileName is provided, it is assumed to be an image. The image data
    will be populated by reading in that file.
    * If both fileName and imgData are provided, then imgData is used to populate the
    image, and fileName is assumed to be the file associated with that data.

    :param fileName: Filename either to load or that corresponds to imgData
    :param imgData: N-Channel numpy image
    :param clearExistingComps: If True, erases all existing components on image load.
      Else, they are retained.
    """
    oldFile = self.srcImgFname
    oldData = self.mainImg.image
    oldComps = self.compMgr.compDf.copy()
    if fileName is not None:
      fileName = Path(fileName).resolve()
    if clearExistingComps:
      self.compMgr.rmComps()
    if imgData is not None:
      self.mainImg.setImage(imgData)
    else:
      self.mainImg.setImage(fileName)
    self.srcImgFname = fileName
    self.focusedImg.resetImage()
    self.mainImg.plotItem.vb.autoRange()
    if self.estBoundsOnStart:
      self.estimateBoundaries()
    yield
    self.resetMainImg(oldFile, oldData, clearExistingComps)
    if clearExistingComps:
      # Old comps were cleared, so put them back
      self.compMgr.addComps(oldComps)

  def loadLayout(self, layoutName: str):
    layoutFilename = LAYOUTS_DIR/f'{layoutName}.dockstate'
    self.restoreState(attemptFileLoad(layoutFilename))

  def saveLayout(self, layoutName: str=None, allowOverwriteDefault=False):
    dockStates = self.saveState().data()
    saveToFile(dockStates, LAYOUTS_DIR/f'{layoutName}.dockstate',
               allowOverwriteDefault=allowOverwriteDefault)
    self.sigLayoutSaved.emit()

  def importQuickLoaderProfile(self, profileSrc: Union[str, dict]):
    if isinstance(profileSrc, str):
      profileSrc = {FR_SINGLETON.quickLoader.name: profileSrc}

    loads = ['Image', 'Annotations', 'Layout']
    fns = [self.resetMainImg, self.loadCompList, self.loadLayoutActionTriggered]
    errSettings = []

    for load, fn in zip(loads, fns):
      try:
        val: Any = profileSrc.pop(load, None)
        if val is not None:
          fn(val)
      except Exception as ex:
        errSettings.append(f'{load}: {ex}')

    if profileSrc:
      # Unclaimed arguments
      try:
        FR_SINGLETON.quickLoader.buildFromUserProfile(profileSrc)
      except Exception as ex:
        errSettings.append(f'{"Quick Loader"}: {ex}')

    if len(errSettings) > 0:
      err = FRAppIOError('The following settings were not loaded (shown as <setting>: <exception>)\n'
                         + "\n\n".join(errSettings))
      raiseErrorLater(err)

  def exportCompList(self, outFname: Union[str, Path]):
    self.compExporter.prepareDf(self.compMgr.compDf, self.compDisplay.displayedIds,
                                self.srcImgFname)
    self.compExporter.exportByFileType(outFname)
    self.hasUnsavedChanges = False

  def exportLabeledImg(self, outFname: str=None):
    self.compExporter.prepareDf(self.compMgr.compDf, self.compDisplay.displayedIds)
    return self.compExporter.exportLabeledImg(self.mainImg.image.shape, outFname)

  def loadCompList(self, inFname: str, loadType=FR_ENUMS.COMP_ADD_AS_NEW):
    pathFname = Path(inFname)
    if self.mainImg.image is None:
      raise FRAppIOError('Cannot load components when no main image is set.')
    fType = pathFname.suffix[1:]
    if fType == 'csv':
      newComps = FRComponentIO.buildFromCsv(inFname, self.mainImg.image.shape)
    elif fType == 's3apkl':
      # Operation may take a long time, but we don't want to start the wait cursor until
      # after dialog selection
      newComps = FRComponentIO.buildFromPkl(inFname, self.mainImg.image.shape)
    else:
      raise FRAppIOError(f'Extension {fType} is not recognized. Must be one of: csv, s3apkl')
    self.compMgr.addComps(newComps, loadType)

  def showNewCompAnalytics(self):
    self._check_plotStages(self.mainImg)

  def showModCompAnalytics(self):
    self._check_plotStages(self.focusedImg)

  # ---------------
  # MISC CALLBACKS
  # ---------------
  def _recordCompChange(self):
    self.hasUnsavedChanges = True

  @FR_SINGLETON.actionStack.undoable('Create New Comp', asGroup=True)
  def add_focusComp(self, newComps: df):
    self.compMgr.addComps(newComps)
    # Make sure index matches ID before updating current component
    newComps = newComps.set_index(REQD_TBL_FIELDS.INST_ID, drop=False)
    # Focus is performed by comp table
    if self.isVisible():
      self.compTbl.setAs(newComps.index)
    self.changeFocusedComp(newComps)

  @staticmethod
  def _check_plotStages(img: FREditableImg):
    proc = img.curProcessor.processor
    if proc.result is None:
      raise FRAlgProcessorError('Analytics can only be shown after the algorithm'
                                ' was run.')
    proc.plotStages(ignoreDuplicateResults=True)


  # ---------------
  # MENU CALLBACKS
  # ---------------
  def openImgActionTriggered(self):
    fileFilter = "Image Files (*.png; *.tif; *.jpg; *.jpeg; *.bmp; *.jfif);; All files(*.*)"
    fname = popupFilePicker(self, 'Select Main Image', fileFilter)
    if fname is not None:
      with BusyCursor():
        self.resetMainImg(fname)

  def loadLayoutActionTriggered(self, layoutName):
    self.loadLayout(layoutName)

  def populateLoadLayoutOptions(self):
    layoutGlob = LAYOUTS_DIR.glob('*.dockstate')
    addDirItemsToMenu(self.menuLayout, layoutGlob, self.loadLayoutActionTriggered)

  def saveLayoutActionTriggered(self):
    outName = dialogGetSaveFileName(self, 'Layout Name')
    if outName is None or outName == '':
      return
    self.saveLayout(outName)

  @staticmethod
  def paramEditorLoadActTriggered(objForMenu: FRParamEditor, nameToLoad: str) -> Optional[dict]:
    with BusyCursor():
      return objForMenu.loadParamState(nameToLoad)

  @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_EXPORT_COMP_LIST)
  def exportCompListActionTriggered(self):
    fileDlg = QtWidgets.QFileDialog()
    # TODO: Delegate this to the exporter. Make a function that makes the right file filter,
    #   and calls the right exporter function after the filename is retrieved.
    fileFilter = "CSV Files (*.csv)"
    fname, _ = fileDlg.getSaveFileName(self, 'Select Save File', '', fileFilter)
    if len(fname) > 0:
      self.exportCompList(fname)

  def exportLabelImgActionTriggered(self):
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
    fileFilter = "Label Mask Image (*.png; *.tif; *.jpg; *.jpeg; *.bmp; *.jfif);; All files(*.*)"
    fname, _ = fileDlg.getSaveFileName(self, 'Select Save File', '', fileFilter)
    if len(fname) > 0:
      self.exportLabeledImg(fname)


  def loadCompsActionTriggered(self, loadType: _FREnums):
    # TODO: See note about exporting comps. Delegate the filepicker activity to importer
    fileFilter = "CSV Files (*.csv)"
    fname = popupFilePicker(self, 'Select Load File', fileFilter)
    if fname is None:
      return
    self.loadCompList(fname, loadType)

  def newCompAnalyticsActTriggered(self):
    self.showNewCompAnalytics()

  # ---------------
  # BUTTON CALLBACKS
  # ---------------
  # Push buttons
  def clearRegionBtnClicked(self):
    self.clearFocusedRegion()

  def resetRegionBtnClicked(self):
    self.resetFocusedRegion()

  @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_ACCEPT_REGION)
  def acceptRegionBtnClicked(self):
    self.acceptFocusedRegion()

  @disableAppDuringFunc
  @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_ESTIMATE_BOUNDARIES)
  def estimateBoundariesBtnClicked(self):
    self.estimateBoundaries()

  @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_CLEAR_BOUNDARIES)
  def clearBoundariesClicked(self):
    self.clearBoundaries()
  # ---------------
  # CUSTOM UI ELEMENT CALLBACKS
  # ---------------
  @FR_SINGLETON.actionStack.undoable('Change Focused Component')
  def changeFocusedComp(self, newComps: df, forceKeepLastChange=False):
    oldSer = self.focusedImg.compSer.copy()
    oldImg = self.focusedImg.image
    if len(newComps) == 0:
      return
    # TODO: More robust scenario if multiple comps are in the dataframe
    #   For now, just use the last in the selection. This is so that if multiple
    #   components are selected in a row, the most recently selected is always
    #   the current displayed.
    newComp: pd.Series = newComps.iloc[-1,:]
    newCompId = newComp[REQD_TBL_FIELDS.INST_ID]
    self.compDisplay.regionPlot.focusById([newCompId])
    mainImg = self.mainImg.image
    self.focusedImg.updateAll(mainImg, newComp)
    self.curCompIdLbl.setText(f'Component ID: {newCompId}')
    # Nothing happened since the last component change, so just replace it instead of
    # adding a distinct action to the buffer queue
    stack = FR_SINGLETON.actionStack
    if not forceKeepLastChange and stack.undoDescr == 'Change Focused Component':
      stack.actions.pop()
    yield
    if oldImg is not None and len(oldSer.loc[REQD_TBL_FIELDS.VERTICES]) > 0:
      self.changeFocusedComp(oldSer.to_frame().T, forceKeepLastChange=True)
    else:
      self.focusedImg.resetImage()
