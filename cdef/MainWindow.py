# -*- coding: utf-8 -*-

import sys
from functools import partial
from os.path import join
from pathlib import Path
from typing import Callable

import pandas as pd
import pyqtgraph as pg
from pandas import DataFrame as df
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui

from cdef.generalutils import resolveAuthorName
from cdef.structures import FRCompIOError
from cdef.tablemodel import FRComponentIO
from .frgraphics.annotator_ui import FRAnnotatorUI
from .frgraphics.graphicsutils import applyWaitCursor, dialogSaveToFile, \
  addDirItemsToMenu, \
  attemptLoadSettings, popupFilePicker, disableAppDuringFunc
from .frgraphics.parameditors import FRParamEditor, FR_SINGLETON
from .projectvars.constants import FR_CONSTS
from .projectvars.constants import LAYOUTS_DIR, TEMPLATE_COMP as TC
from .projectvars.enums import FR_ENUMS
from .tablemodel import FRComponentMgr, makeCompDf
from .tableviewproxy import CompDisplayFilter, CompSortFilter

Slot = QtCore.pyqtSlot
Signal = QtCore.pyqtSignal

# Configure pg to correctly read image dimensions
pg.setConfigOptions(imageAxisOrder='row-major')

@FR_SINGLETON.registerClass(FR_CONSTS.CLS_ANNOTATOR)
class MainWindow(FRAnnotatorUI):
  """
  Top-level widget for producing component bounding boxes from an input image.
  """
  # Alerts GUI that a layout (either new or overwriting old) was saved
  sigLayoutSaved = Signal()

  def __init__(self, startImgFpath: str = None, startAnnFpath: str = None, authorName: str = None):
    super().__init__()
    
    # ---------------
    # DATA ATTRIBUTES
    # ---------------
    self.mainImgFpath = None
    if startImgFpath is not None:
      # Make sure to simplify the incoming path
      self.mainImgFpath = str(Path(startImgFpath).resolve())
      startImgFpath = self.mainImgFpath
    self.hasUnsavedChanges = False

    self.statBar = QtWidgets.QStatusBar(self)
    self.setStatusBar(self.statBar)
    authorName = resolveAuthorName(authorName)
    if authorName is None:
      sys.exit('No author name provided and no default author exists. Exiting.\n'
               'To start without error, provide an author name explicitly.')
    FR_SINGLETON.annotationAuthor = authorName
    self.statBar.showMessage(FR_SINGLETON.annotationAuthor)

    # Flesh out pg components
    # ---------------
    # MAIN IMAGE
    # ---------------
    self.mainImg.sigComponentCreated.connect(self._add_focusComp)
    self.mainImg.setImage(startImgFpath)
    # ---------------
    # FOCUSED IMAGE
    # ---------------

    # ---------------
    # COMPONENT MANAGER
    # ---------------
    self.compMgr = FRComponentMgr()
    self.compMgr.sigCompsChanged.connect(self._recordCompChange)

    # Allow filtering/sorting
    self.sortFilterProxy = CompSortFilter(self.compMgr, self)

    self.compTbl.setModel(self.sortFilterProxy)

    # ---------------
    # COMPONENT DISPLAY FILTER
    # ---------------
    self.compDisplay = CompDisplayFilter(self.compMgr, self.mainImg, self.compTbl,
                                         self)

    self.mainImg.imgItem.sigImageChanged.connect(self.clearBoundaries)
    self.compDisplay.sigCompsSelected.connect(self.updateCurComp)

    # ---------------
    # UI ELEMENT SIGNALS
    # ---------------
    # Buttons
    self.openImgAct.triggered.connect(self.openImgActionTriggered)
    self.clearRegionBtn.clicked.connect(self.clearRegionBtnClicked)
    self.resetRegionBtn.clicked.connect(self.resetRegionBtnClicked)
    self.acceptRegionBtn.clicked.connect(self.acceptRegionBtnClicked)

    # Same with estimating boundaries
    if (startImgFpath is not None
        and self.estBoundsOnStart):
      self.estimateBoundaries()

    if startAnnFpath is not None:
      self.loadCompsActionTriggered(FR_ENUMS.COMP_ADD_AS_NEW, startAnnFpath)



    # Menu options
    # FILE
    self.saveLayout.triggered.connect(self.saveLayoutActionTriggered)
    self.sigLayoutSaved.connect(self.populateLoadLayoutOptions)

    self.exportCompList.triggered.connect(self.exportCompListActionTriggered)
    self.exportLabelImg.triggered.connect(self.exportLabelImgActionTriggered)
    self.loadComps_merge.triggered.connect(lambda: self.loadCompsActionTriggered(FR_ENUMS.COMP_ADD_AS_MERGE))
    self.loadComps_new.triggered.connect(lambda: self.loadCompsActionTriggered(FR_ENUMS.COMP_ADD_AS_NEW))

    # SETTINGS
    self.createSettingsMenus()

    # ---------------
    # LOAD LAYOUT OPTIONS
    # ---------------
    self.populateLoadLayoutOptions()
    # Start with docks in default position, hide error if default file doesn't exist
    self.loadLayoutActionTriggered('Default', showError=False)

  # -----------------------------
  # MainWindow CLASS FUNCTIONS
  # -----------------------------
  @FR_SINGLETON.generalProps.registerProp(FR_CONSTS.PROP_EST_BOUNDS_ON_START)
  def estBoundsOnStart(self): pass

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


  def createSettingsMenus(self):
    for editor, name in zip(FR_SINGLETON.editors, FR_SINGLETON.editorNames): \
        #type: FRParamEditor, str
      menu = QtWidgets.QMenu(name, self)
      editAct = QtWidgets.QAction('Edit ' + name, self)
      menu.addAction(editAct)
      menu.addSeparator()
      editAct.triggered.connect(editor.show)
      loadFunc = partial(self.genericLoadActionTriggered, editor)
      populateFunc = partial(self.genericPopulateMenuOptions, editor, menu, loadFunc)
      editor.sigParamStateCreated.connect(populateFunc)
      # Initialize default menus
      populateFunc()
      self.menuSettings.addMenu(menu)

  @Slot(object)
  def _recordCompChange(self):
    self.hasUnsavedChanges = True

  @Slot(object)
  def _add_focusComp(self, newComp):
    self.compMgr.addComps(newComp)
    # Make sure index matches ID before updating current component
    newComp = newComp.set_index(TC.INST_ID, drop=False)
    # Set this component as active in the focused view
    self.updateCurComp(newComp)

  # ---------------
  # MENU CALLBACKS
  # ---------------

  @Slot()
  @applyWaitCursor
  def openImgActionTriggered(self):
    fileFilter = "Image Files (*.png; *.tif; *.jpg; *.jpeg; *.bmp; *.jfif);; All files(*.*)"
    fname = popupFilePicker(self, 'Select Main Image', fileFilter)

    if fname is not None:
      self.compMgr.rmComps()
      self.mainImg.setImage(fname)
      self.mainImgFpath = fname
      self.compImg.resetImage()
      if self.estBoundsOnStart:
        self.estimateBoundaries()

  def populateLoadLayoutOptions(self):
    layoutGlob = join(LAYOUTS_DIR, '*.dockstate')
    addDirItemsToMenu(self.menuLayout, layoutGlob, self.loadLayoutActionTriggered)

  @Slot(str)
  def loadLayoutActionTriggered(self, layoutName, showError=True):
    layoutFilename = join(LAYOUTS_DIR, f'{layoutName}.dockstate')
    dockStates = attemptLoadSettings(layoutFilename, showErrorOnFail=showError)
    if dockStates is not None:
      self.restoreState(dockStates)

  @Slot()
  def saveLayoutActionTriggered(self):
    dockStates = self.saveState()
    dialogSaveToFile(self, dockStates, 'Layout Name', LAYOUTS_DIR, 'dockstate')
    self.sigLayoutSaved.emit()

  @Slot()
  def exportCompListActionTriggered(self):
    onlyExportFiltered = self.compMgr.exportOnlyVis
    if onlyExportFiltered:
      exportIds = self.compDisplay.displayedIds
    else:
      exportIds = FR_ENUMS.COMP_EXPORT_ALL
    fileDlg = QtWidgets.QFileDialog()
    # TODO: Delegate this to the exporter. Make a function that makes the right file filter,
    #   and calls the right exporter function after the filename is retrieved.
    fileFilter = "CSV Files (*.csv)"
    fname, _ = fileDlg.getSaveFileName(self, 'Select Save File', '', fileFilter)
    if len(fname) > 0:
      exporter = FRComponentIO(self.compMgr.compDf, self.mainImgFpath, exportIds)
      exporter.exportCsv(fname)
      self.hasUnsavedChanges = False

  @Slot()
  def exportLabelImgActionTriggered(self):
    """
    # Note -- These three functions will be a single dialog with options
    # for each requested parameter. It will look like the TableFilterEditor dialog.
    types: List[CompParams] = getTypesFromUser()
    outFile = getOutFileFromUser()
    exportLegend = getExpLegendFromUser()
    """
    onlyExportFiltered = self.compMgr.exportOnlyVis
    if onlyExportFiltered:
      exportIds = self.compDisplay.displayedIds
    else:
      exportIds = FR_ENUMS.COMP_EXPORT_ALL
    fileDlg = QtWidgets.QFileDialog()
    # TODO: Delegate this to the exporter. Make a function that makes the right file filter,
    #   and calls the right exporter function after the filename is retrieved.
    fileFilter = "Label Mask Image (*.png; *.tif; *.jpg; *.jpeg; *.bmp; *.jfif);; All files(*.*)"
    fname, _ = fileDlg.getSaveFileName(self, 'Select Save File', '', fileFilter)
    if len(fname) > 0:
      exporter = FRComponentIO(self.compMgr.compDf, self.mainImgFpath, exportIds)
      exporter.exportLabeledImg(self.mainImg.image.shape, fname)


  def loadCompsActionTriggered(self, loadType=FR_ENUMS.COMP_ADD_AS_NEW, fname: str=None):
    # TODO: See note about exporting comps. Delegate the filepicker activity to importer
    if fname is None:
      fileFilter = "CSV Files (*.csv)"
      fname = popupFilePicker(self, 'Select Load File', fileFilter)
    if fname is None:
      return
    pathFname = Path(fname)
    fType = pathFname.suffix[1:]
    if fType == 'csv':
      newComps, errMsg = FRComponentIO.buildFromCsv(fname, self.mainImg.image.shape)
    elif fType == 'cdefpkl':
      # Operation may take a long time, but we don't want to start the wait cursor until
      # after dialog selection
      newComps, errMsg = FRComponentIO.buildFromPkl(fname, self.mainImg.image.shape)
    else:
      raise FRCompIOError(f'Extension {fType} is not recognized. Must be one of: csv, cdefpkl')
    if errMsg is not None:
      # Something went wrong. Inform the user.
      fullErrMsg = f'Failed to import components:\n{errMsg}'
      QtWidgets.QMessageBox().information(self, 'Error During Import', fullErrMsg)
    else:
      self.compMgr.addComps(newComps, loadType)

  @staticmethod
  def genericPopulateMenuOptions(objForMenu: FRParamEditor, winMenu: QtWidgets.QMenu, triggerFn: Callable):
    addDirItemsToMenu(winMenu,
                      join(objForMenu.saveDir, f'*.{objForMenu.fileType}'),
                      triggerFn)

  @staticmethod
  def genericLoadActionTriggered(objForMenu: FRParamEditor, nameToLoad: str):
    dictFilename = join(objForMenu.saveDir, f'{nameToLoad}.{objForMenu.fileType}')
    loadDict = attemptLoadSettings(dictFilename)
    if loadDict is None:
      return
    objForMenu.loadState(loadDict)
    objForMenu.applyBtnClicked()

  # ---------------
  # BUTTON CALLBACKS
  # ---------------
  # Push buttons
  @Slot()
  def clearRegionBtnClicked(self):
    # Reset drawn comp vertices to nothing
    # Only perform action if image currently exists
    if self.compImg.imgItem.image is None:
      return
    self.compImg.updateRegionFromVerts(None)

  @Slot()
  def resetRegionBtnClicked(self):
    # Reset drawn comp vertices to nothing
    # Only perform action if image currently exists
    if self.compImg.imgItem.image is None:
      return
    self.compImg.updateRegionFromVerts(self.compImg.compSer[TC.VERTICES])

  @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_ACCEPT_REGION)
  @Slot()
  def acceptRegionBtnClicked(self):
    self.compImg.saveNewVerts()
    modifiedComp = self.compImg.compSer
    self.compMgr.addComps(modifiedComp.to_frame().T, addtype=FR_ENUMS.COMP_ADD_AS_MERGE)

  @disableAppDuringFunc
  @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_ESTIMATE_BOUNDARIES)
  def estimateBoundaries(self):
    compVertices = self.mainImg.procCollection.curProcessor.globalCompEstimate()
    components = makeCompDf(len(compVertices))
    components[TC.VERTICES] = compVertices
    self.compMgr.addComps(components)

  @Slot()
  @applyWaitCursor
  @FR_SINGLETON.shortcuts.registerMethod(FR_CONSTS.SHC_CLEAR_BOUNDARIES)
  def clearBoundaries(self):
    self.compMgr.rmComps()

  # ---------------
  # CUSTOM UI ELEMENT CALLBACKS
  # ---------------
  @Slot(object)
  @applyWaitCursor
  def updateCurComp(self, newComps: df):
    if len(newComps) == 0:
      return
    # TODO: More robust scenario if multiple comps are in the dataframe
    #   For now, just use the last in the selection. This is so that if multiple
    #   components are selected in a row, the most recently selected is always
    #   the current displayed.
    newComps: pd.Series = newComps.iloc[-1,:]
    newCompId = newComps[TC.INST_ID]
    self.compDisplay.regionPlots.focusById([newCompId])
    mainImg = self.mainImg.image
    self.compImg.updateAll(mainImg, newComps)
    self.curCompIdLbl.setText(f'Component ID: {newCompId}')

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
  app = pg.mkQApp()
  win = MainWindow()
  app.exec()