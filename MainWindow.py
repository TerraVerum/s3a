# -*- coding: utf-8 -*-

import os
from os.path import join

import numpy as np
import pyqtgraph as pg
from pandas import DataFrame as df
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui, uic

from ABGraphics.parameditors import SchemeEditor, TableFilterEditor, RegionControlsEditor
from ABGraphics.utils import applyWaitCursor, dialogSaveToFile, addDirItemsToMenu, attemptLoadSettings
from CompDisplayFilter import CompDisplayFilter
from constants import RegionControlsEditorValues as RCEV
from constants import SCHEMES_DIR, LAYOUTS_DIR, TEMPLATE_COMP as TC
from processing import getBwComps, getVertsFromBwComps, getClippedBbox
from tablemodel import ComponentMgr as ComponentMgr
from tablemodel import makeCompDf

Slot = QtCore.pyqtSlot
Signal = QtCore.pyqtSignal

# Configure pg to correctly read image dimensions
pg.setConfigOptions(imageAxisOrder='row-major')

class MainWindow(QtWidgets.QMainWindow):
  # Alerts GUI that a layout (either new or overwriting old) was saved
  sigLayoutSaved = Signal()

  def __init__(self, startImgFpath=None):
    super().__init__()
    uiPath = os.path.dirname(os.path.abspath(__file__))
    uiFile = os.path.join(uiPath, 'imgAnnotator.ui')
    uic.loadUi(uiFile, self)

    # Flesh out pg components
    # ---------------
    # MAIN IMAGE
    # ---------------
    self.mainImg.setImage(startImgFpath)
    self.mainImg.imgItem.sigClicked.connect(self.mainImgItemClicked)

    # ---------------
    # LOAD LAYOUT OPTIONS
    # ---------------
    self.populateLoadLayoutOptions()
    # Start with docks in default position
    self.loadLayoutActionTriggered('Default')

    # ---------------
    # COMPONENT MANAGER
    # ---------------
    self.compMgr = ComponentMgr()

    self.compTbl.setModel(self.compMgr)

    # ---------------
    # COMPONENT DISPLAY FILTER
    # ---------------
    # TODO: Add filter widget for displaying only part of component data
    self.filterEditor = TableFilterEditor()
    self.compDisplay = CompDisplayFilter(self.compMgr, self.mainImg, self.compTbl, self.filterEditor)

    self.mainImg.imgItem.sigImageChanged.connect(self.clearBoundsBtnClicked)
    self.compDisplay.sigCompClicked.connect(self.updateCurComp)


    # ---------------
    # LOAD SCHEME OPTIONS
    # ---------------
    self.scheme = SchemeEditor(self)
    self.populateSchemeOptions()
    # Attach scheme to all UI children
    self.compImg.setScheme(self.scheme)
    #Component.setScheme(self.scheme)
    CompDisplayFilter.setScheme(self.scheme)

    # ---------------
    # LOAD REGION EDIT CONTROLS
    # ---------------
    self.regCtrlEditor = RegionControlsEditor()

    # ---------------
    # UI ELEMENT SIGNALS
    # ---------------
    # Buttons
    self.newImgBtn.clicked.connect(self.newImgBtnClicked)
    self.estBoundsBtn.clicked.connect(self.estBoundsBtnClicked)
    self.clearBoundsBtn.clicked.connect(self.clearBoundsBtnClicked)
    self.filterBtn.clicked.connect(self.filterBtnClicked)
    self.clearRegionBtn.clicked.connect(self.clearRegionBtnClicked)
    self.resetRegionBtn.clicked.connect(self.resetRegionBtnClicked)
    self.acceptRegionBtn.clicked.connect(self.acceptRegionBtnClicked)

    # Radio buttons
    self.regionRadioBtnGroup.buttonClicked.connect(self.regionTypeChanged)

    # Dropdowns
    self.addRmCombo.currentIndexChanged.connect(self.addRmComboChanged)

    # Checkboxes
    self.allowEditsChk.stateChanged.connect(self.allowEditsChkChanged)


    # Edit fields
    self.regCtrlEditor[RCEV.SEED_THRESH].sigValueChanged.connect(self.seedThreshChanged)
    # Note: This signal must be false-triggered on startup to propagate
    # the field's initial value
    self.regCtrlEditor[RCEV.SEED_THRESH].sigValueChanged.emit(None, None)

    # Menu options
    # FILE
    self.saveLayout.triggered.connect(self.saveLayoutActionTriggered)
    self.sigLayoutSaved.connect(self.populateLoadLayoutOptions)
    self.saveComps.triggered.connect(self.saveCompsActionTriggered)
    self.loadComps_merge.triggered.connect(lambda: self.loadCompsActionTriggered('merge'))
    self.loadComps_add.triggered.connect(lambda: self.loadCompsActionTriggered('add'))

    # SETTINGS
    self.compEditCtrls.triggered.connect(self.regCtrlEditor.show)

    self.editScheme.triggered.connect(self.scheme.show)
    self.scheme.sigParamStateCreated.connect(self.populateSchemeOptions)
    # When a new scheme is created, switch to that scheme
    self.scheme.sigParamStateCreated.connect(self.loadSchemeActionTriggered)

  # -----------------------------
  # MainWindow CLASS FUNCTIONS
  # -----------------------------
  def closeEvent(self, ev):
    # Clean up all child windows, which could potentially be left open
    self.filterEditor.close()
    self.scheme.close()
    self.regCtrlEditor.close()

  # ---------------
  # MENU CALLBACKS
  # ---------------
  def populateLoadLayoutOptions(self):
    layoutGlob = join(LAYOUTS_DIR, '*.dockstate')
    addDirItemsToMenu(self.loadLayout, layoutGlob, self.loadLayoutActionTriggered)

  @Slot(str)
  def loadLayoutActionTriggered(self, layoutName):
    layoutFilename = join(LAYOUTS_DIR, f'{layoutName}.dockstate')
    dockStates = attemptLoadSettings(layoutFilename)
    if dockStates is not None:
      self.restoreState(dockStates)

  @Slot()
  def saveLayoutActionTriggered(self):
    dockStates = self.saveState()
    dialogSaveToFile(self, dockStates, 'Layout Name', LAYOUTS_DIR, 'dockstate')
    self.sigLayoutSaved.emit()

  @Slot()
  def saveCompsActionTriggered(self):
    fileDlg = QtWidgets.QFileDialog()
    fileFilter = "CSV Files (*.csv)"
    fname, _ = fileDlg.getSaveFileName(self, 'Select Save File', '', fileFilter)
    if len(fname) > 0:
      self.compMgr.csvExport(fname)

  def loadCompsActionTriggered(self, loadType='add'):
    fileDlg = QtWidgets.QFileDialog()
    fileFilter = "CSV Files (*.csv)"
    fname, _ = fileDlg.getOpenFileName(self, 'Select Load File', '', fileFilter)
    if len(fname) > 0:
      # Operation may take a long time, but we don't want to start the wait cursor until
      # after dialog selection
      applyWaitCursor(self.compMgr.csvImport)(fname, loadType)

  # noinspection PyUnusedLocal
  @Slot(str)
  def populateSchemeOptions(self, newSchemeName=None):
    # We don't want all menu children to be removed, since this would also remove the 'add scheme' and
    # separator options. So, do this step manually. Remove all actions after the separator
    encounteredSep = False
    for ii, action in enumerate(self.appearanceMenu.children()):
      if encounteredSep:
        self.appearanceMenu.removeAction(action)
      elif action.isSeparator():
        encounteredSep = True
    addDirItemsToMenu(self.appearanceMenu, join(SCHEMES_DIR, '*.scheme'),
                      self.loadSchemeActionTriggered, removeExistingChildren=False)

  @Slot(str)
  def loadSchemeActionTriggered(self, schemeName):
    schemeFilename = join(SCHEMES_DIR, f'{schemeName}.scheme')
    schemeDict = attemptLoadSettings(schemeFilename)
    if schemeDict is None:
      return
    self.scheme.loadScheme(schemeDict)

    QtWidgets.QMessageBox().information(self, 'Scheme Updated',
                                        'Scheme updated. Changes will take effect in future operations.',
                                        QtGui.QMessageBox.Ok)

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
    self.compImg.updateRegion(None)

  @Slot()
  def resetRegionBtnClicked(self):
    # Reset drawn comp vertices to nothing
    # Only perform action if image currently exists
    if self.compImg.compImgItem.image is None:
      return
    self.compImg.updateRegion(self.compImg.compSer[TC.VERTICES.name].squeeze())

  @Slot()
  def acceptRegionBtnClicked(self):
    self.compImg.saveNewVerts()
    self.compMgr.addComps(self.compImg.compSer.to_frame().T, addtype='merge')

  @Slot()
  def newImgBtnClicked(self):
    fileDlg = QtWidgets.QFileDialog()
    fileFilter = "Image Files (*.png; *.tif; *.jpg; *.jpeg; *.bmp)"
    fname, _ = fileDlg.getOpenFileName(self, 'Select Main Image', '', fileFilter)

    if len(fname) > 0:
      self.mainImg.setImage(fname)

  @Slot()
  @applyWaitCursor
  def estBoundsBtnClicked(self):
    self.compMgr.rmComps()
    compVertices = getVertsFromBwComps(getBwComps(self.mainImg.image))
    components = makeCompDf(len(compVertices))
    components[TC.VERTICES.name] = compVertices
    self.compMgr.addComps(components)

  @Slot()
  @applyWaitCursor
  def clearBoundsBtnClicked(self):
    self.compMgr.rmComps()

  @Slot()
  @applyWaitCursor
  def filterBtnClicked(self):
    self.filterEditor.show()

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

  # ---------------
  # TEXT EDIT CALLBACKS
  # ---------------
  @Slot()
  def seedThreshChanged(self):
    self.compImg.seedThresh = self.regCtrlEditor[RCEV.SEED_THRESH].value()


  # ---------------
  # CUSTOM UI ELEMENT CALLBACKS
  # ---------------
  @Slot(object)
  def mainImgItemClicked(self, xyCoord):
    """
    Forms a box with a center at the clicked location, and passes the box
    edges as vertices for a new component.
    """
    sideLen = self.regCtrlEditor[RCEV.NEW_COMP_SZ].value()
    vertBox = np.vstack((xyCoord, xyCoord))
    vertBox = getClippedBbox(self.mainImg.image.shape, vertBox, sideLen)
    # Create square from bounding box
    compVerts = [
      [vertBox[0, 0], vertBox[0, 1]],
      [vertBox[1, 0], vertBox[0, 1]],
      [vertBox[1, 0], vertBox[1, 1]],
      [vertBox[0, 0], vertBox[1, 1]]
    ]
    compVerts = np.vstack(compVerts)

    newComp = makeCompDf()
    newComp[TC.VERTICES.name] = [compVerts]
    self.compMgr.addComps(newComp)


  @Slot(object)
  @applyWaitCursor
  def updateCurComp(self, newComp: df):
    mainImg = self.mainImg.image
    margin = self.regCtrlEditor[RCEV.MARGIN].value()
    segThresh = self.regCtrlEditor[RCEV.SEG_THRESH].value()
    prevComp = self.compImg.compSer
    rmPrevComp = self.compImg.updateAll(mainImg, newComp, margin, segThresh)
    # If all old vertices were deleted AND we switched images, signal deletion
    # for the previous focused component
    if rmPrevComp:
      self.compMgr.rmComps(prevComp[TC.INST_ID.name])

    self.curCompIdLbl.setText(f'Component ID: {newComp[TC.INST_ID.name]}')

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
  app = pg.mkQApp()
  win = MainWindow()
  c = makeCompDf(100)
  c = c.set_index(np.arange(len(c),dtype=int))
  for ii in range(len(c)):
    c.loc[ii,TC.VERTICES.name] = [np.random.randint(100,size=(30,2),dtype=int)]
    c.loc[ii, TC.NOTES.name] = 'test notes'
  # win.compMgr.addComps(c)
  win.show()
  app.exec()