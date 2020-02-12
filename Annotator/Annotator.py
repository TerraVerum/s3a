# -*- coding: utf-8 -*-

import os
from functools import partial
from os.path import join
from typing import Callable

import numpy as np
import pyqtgraph as pg
from pandas import DataFrame as df
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui, uic

from Annotator.constants import AB_CONSTS
from .ABGraphics.parameditors import ConstParamWidget, TableFilterEditor, AB_SINGLETON
from .ABGraphics.graphicsutils import applyWaitCursor, dialogSaveToFile, addDirItemsToMenu, \
  attemptLoadSettings, popupFilePicker, disableAppDuringFunc
from .CompDisplayFilter import CompDisplayFilter, CompSortFilter
from .constants import LAYOUTS_DIR, TEMPLATE_COMP as TC
from .processing import getBwComps, getVertsFromBwComps, growSeedpoint,\
  growBoundarySeeds, pcaReduction
from skimage import morphology
from Annotator.generalutils import nanConcatList, getClippedBbox
from .tablemodel import ComponentMgr as ComponentMgr, AB_ENUMS
from .tablemodel import makeCompDf

Slot = QtCore.pyqtSlot
Signal = QtCore.pyqtSignal

# Configure pg to correctly read image dimensions
pg.setConfigOptions(imageAxisOrder='row-major')

@AB_SINGLETON.registerClass(AB_CONSTS.CLS_ANNOTATOR)
class Annotator(QtWidgets.QMainWindow):
  # Alerts GUI that a layout (either new or overwriting old) was saved
  sigLayoutSaved = Signal()

  def __init__(self, startImgFpath=None):
    super().__init__()
    uiPath = os.path.dirname(os.path.abspath(__file__))
    uiFile = os.path.join(uiPath, 'imgAnnotator.ui')
    baseModule = str(self.__module__).split('.')[0]
    uic.loadUi(uiFile, self, baseModule)
    self.setStatusBar(QtWidgets.QStatusBar())

    # Flesh out pg components
    # ---------------
    # MAIN IMAGE
    # ---------------
    self.mainImg.sigComponentCreated.connect(self._add_focusComp)
    self.mainImg.setImage(startImgFpath)

    # ---------------
    # FOCUSED COMPONENT IMAGE
    # ---------------
    self.compImg.sigEnterPressed.connect(self.acceptRegionBtnClicked)
    self.compImg.sigModeChanged.connect(self.compImgModeChanged)

    # ---------------
    # COMPONENT MANAGER
    # ---------------
    self.compMgr = ComponentMgr()

    # Allow filtering/sorting
    self.sortFilterProxy = CompSortFilter(self.compMgr, self)

    self.compTbl.setModel(self.sortFilterProxy)

    # ---------------
    # COMPONENT DISPLAY FILTER
    # ---------------
    self.compDisplay = CompDisplayFilter(self.compMgr, self.mainImg, self.compTbl)

    self.mainImg.imgItem.sigImageChanged.connect(self.clearBoundaries)
    self.compDisplay.sigCompClicked.connect(self.updateCurComp)

    # ---------------
    # UI ELEMENT SIGNALS
    # ---------------
    # Buttons
    self.openImgAct.triggered.connect(self.openImgActionTriggered)
    self.clearRegionBtn.clicked.connect(self.clearRegionBtnClicked)
    self.resetRegionBtn.clicked.connect(self.resetRegionBtnClicked)
    self.acceptRegionBtn.clicked.connect(self.acceptRegionBtnClicked)

    # Radio buttons
    self.regionRadioBtnGroup.buttonClicked.connect(self.regionTypeChanged)

    # Dropdowns
    self.addRmCombo.currentIndexChanged.connect(self.addRmComboChanged)

    # Checkboxes
    self.allowEditsChk.stateChanged.connect(self.allowEditsChkChanged)


    # Same with estimating boundaries
    if startImgFpath is not None \
       and self.estBoundsOnStart:
      self.estimateBoundaries()

    # Menu options
    # FILE
    self.saveLayout.triggered.connect(self.saveLayoutActionTriggered)
    self.sigLayoutSaved.connect(self.populateLoadLayoutOptions)

    self.saveComps.triggered.connect(self.saveCompsActionTriggered)
    self.loadComps_merge.triggered.connect(lambda: self.loadCompsActionTriggered(
      AB_ENUMS.COMP_ADD_AS_MERGE))
    self.loadComps_new.triggered.connect(lambda: self.loadCompsActionTriggered(
      AB_ENUMS.COMP_ADD_AS_NEW))

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
  # TODO: Move these properties into the class responsible for image processing/etc.
  @AB_SINGLETON.generalProps.registerProp(AB_CONSTS.PROP_EST_BOUNDS_ON_START)
  def estBoundsOnStart(self): pass

  def closeEvent(self, ev):
    # Clean up all editor windows, which could potentially be left open
    AB_SINGLETON.close()

  def createSettingsMenus(self):
    for editor, name in zip(AB_SINGLETON.editors, AB_SINGLETON.editorNames): \
        #type: ConstParamWidget, str
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
  def _add_focusComp(self, newComp):
    self.compMgr.addComps(newComp)
    # Make sure index matches ID before updating current component
    newComp = newComp.set_index(TC.INST_ID, drop=False)
    # Set this component as active in the focused view
    self.updateCurComp(newComp.squeeze())

  # ---------------
  # MENU CALLBACKS
  # ---------------

  @Slot()
  @applyWaitCursor
  def openImgActionTriggered(self):
    fileFilter = "Image Files (*.png; *.tif; *.jpg; *.jpeg; *.bmp)"
    fname = popupFilePicker(self, 'Select Main Image', fileFilter)

    if fname is not None:
      self.compMgr.rmComps()
      self.mainImg.setImage(fname)
      if self.estBoundsOnStart:
        self.estimateBoundaries()

  def populateLoadLayoutOptions(self):
    layoutGlob = join(LAYOUTS_DIR, '*.dockstate')
    addDirItemsToMenu(self.layoutMenu, layoutGlob, self.loadLayoutActionTriggered)

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
  def saveCompsActionTriggered(self):
    onlyExportFiltered = self.compMgr.exportOnlyVis
    if onlyExportFiltered:
      exportIds = self.compDisplay.displayedIds
    else:
      exportIds = AB_ENUMS.COMP_EXPORT_ALL
    fileDlg = QtWidgets.QFileDialog()
    fileFilter = "CSV Files (*.csv)"
    fname, _ = fileDlg.getSaveFileName(self, 'Select Save File', '', fileFilter)
    if len(fname) > 0:
      self.compMgr.csvExport(fname, exportIds)

  def loadCompsActionTriggered(self, loadType=AB_ENUMS.COMP_ADD_AS_NEW):
    fileFilter = "CSV Files (*.csv)"
    fname = popupFilePicker(self, 'Select Load File', fileFilter)
    if fname is not None:
      # Operation may take a long time, but we don't want to start the wait cursor until
      # after dialog selection
      err = applyWaitCursor(self.compMgr.csvImport)(fname, loadType,
                                                    self.mainImg.image.shape)
      if err is not None:
        # Something went wrong. Inform the user.
        errMsg = f'Failed to import components. {type(err)}:\n{err}'
        QtWidgets.QMessageBox().information(self, 'Error During Import', errMsg)

  def genericPopulateMenuOptions(self, objForMenu: ConstParamWidget, winMenu: QtWidgets.QMenu, triggerFn: Callable):
    addDirItemsToMenu(winMenu,
                      join(objForMenu.saveDir, f'*.{objForMenu.fileType}'),
                      triggerFn)

  def genericLoadActionTriggered(self, objForMenu: ConstParamWidget, nameToLoad: str):
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
    if self.compImg.compImgItem.image is None:
      return
    self.compImg.updateRegionFromVerts(None)

  @Slot()
  def resetRegionBtnClicked(self):
    # Reset drawn comp vertices to nothing
    # Only perform action if image currently exists
    if self.compImg.compImgItem.image is None:
      return
    self.compImg.updateRegionFromVerts(self.compImg.compSer[TC.VERTICES].squeeze())

  @Slot()
  def acceptRegionBtnClicked(self):
    self.compImg.saveNewVerts()
    modifiedComp = self.compImg.compSer
    self.compMgr.addComps(modifiedComp.to_frame().T, addtype=AB_ENUMS.COMP_ADD_AS_MERGE)

  @disableAppDuringFunc
  @AB_SINGLETON.shortcuts.registerMethod(AB_CONSTS.SHC_ESTIMATE_BOUNDARIES)
  def estimateBoundaries(self):
    compVertices = getVertsFromBwComps(getBwComps(self.mainImg.image, self.minCompSz))
    components = makeCompDf(len(compVertices))
    components[TC.VERTICES] = compVertices
    self.compMgr.addComps(components)

  @Slot()
  @applyWaitCursor
  @AB_SINGLETON.shortcuts.registerMethod(AB_CONSTS.SHC_CLEAR_BOUNDARIES)
  def clearBoundaries(self):
    self.compMgr.rmComps()

  # ---------------
  # CHECK BOX CALLBACKS
  # ---------------
  @Slot()
  def allowEditsChkChanged(self):
    self.compImg.clickable = self.allowEditsChk.isChecked()

  # ---------------
  # RADIO BUTTON CALLBACKS
  # ---------------
  @Slot()
  def regionTypeChanged(self):
    regionType = self.regionRadioBtnGroup.checkedButton().text()
    self.compImg.regionType = regionType.lower()

  # ---------------
  # COMBO BOX CALLBACKS
  # ---------------
  @Slot(int)
  def addRmComboChanged(self):
    curTxt = self.addRmCombo.currentText()
    self.compImg.inAddMode = curTxt == 'Add'

  @Slot(bool)
  def compImgModeChanged(self, newMode):
    self.addRmCombo.blockSignals(True)
    self.addRmCombo.setCurrentIndex(not newMode)
    self.addRmCombo.blockSignals(False)

  # ---------------
  # CUSTOM UI ELEMENT CALLBACKS
  # ---------------
  @Slot(object)
  @applyWaitCursor
  def updateCurComp(self, newComp: df):
    mainImg = self.mainImg.image
    prevComp = self.compImg.compSer
    rmPrevComp = self.compImg.updateAll(mainImg, newComp)
    # If all old vertices were deleted AND we switched images, signal deletion
    # for the previous focused component
    if rmPrevComp:
      self.compMgr.rmComps(prevComp[TC.INST_ID])
    self.curCompIdLbl.setText(f'Component ID: {newComp[TC.INST_ID]}')

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
  app = pg.mkQApp()
  win = Annotator()
  win.show()
  app.exec()