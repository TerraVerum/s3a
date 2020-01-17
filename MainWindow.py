# -*- coding: utf-8 -*-

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui, uic
Slot = QtCore.pyqtSlot
Signal = QtCore.pyqtSignal

import numpy as np
from pandas import DataFrame as df
from PIL import Image

from processing import getBwComps, getVertsFromBwComps, getClippedBbox
from ABGraphics.utils import applyWaitCursor, dialogSaveToFile, addDirItemsToMenu, attemptLoadSettings
from ABGraphics.parameditors import SchemeEditor, TableFilterEditor, RegionControlsEditor
from dataTable import CompTableModel, makeCompDf
#from component import Component, ComponentMgr, CompDisplayFilter
from dataTable import DataComponentMgr as ComponentMgr
from dataModelComponent import CompDisplayFilter
from constants import SCHEMES_DIR, LAYOUTS_DIR, TEMPLATE_COMP as TC
from constants import RegionControlsEditorValues as RCEV
from ABGraphics.clickables import ClickableImageItem

import os
from os.path import join

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
    imgArray = None
    if startImgFpath is not None:
      imgArray = np.array(Image.open(startImgFpath))
    item = ClickableImageItem(imgArray)
    # Ensure image will remain in background of window
    item.setZValue(-100)
    self.mainImg.addItem(item)
    self.mainImg.setAspectLocked(True)
    self.mainImgItem = item
    item.sigClicked.connect(self.mainImgItemClicked)

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

    self.mainImgItem.sigImageChanged.connect(self.compDisplay.resetCompBounds)
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
    self.saveLayout.triggered.connect(self.saveLayoutActionTriggered)
    self.sigLayoutSaved.connect(self.populateLoadLayoutOptions)

    self.newScheme.triggered.connect(self.scheme.show)

    self.compEditCtrls.triggered.connect(self.regCtrlEditor.show)

    # Scheme editor
    self.scheme.sigParamStateCreated.connect(self.populateSchemeOptions)
    # When a new scheme is created, switch to that scheme
    self.scheme.sigParamStateCreated.connect(self.loadSchemeActionTriggered)

  # -----------------------------
  # MainWindow CLASS FUNCTIONS
  # -----------------------------

  def resetMainImg(self, newIm: np.array):
    self.mainImgItem.setImage(newIm)
    self.compMgr.rmComps('all')


  # -----------------------------
  # SIGNAL CALLBACK FUNCTIONS
  # -----------------------------
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
    self.compImg.updateRegion(self.compImg.comp[TC.VERTICES.name].squeeze())

  @Slot()
  def acceptRegionBtnClicked(self):
    self.compImg.saveNewVerts()
    self.compMgr.addComps(self.compImg.comp, addtype='merge')

  @Slot()
  def newImgBtnClicked(self):
    fileDlg = QtWidgets.QFileDialog()
    fileFilter = "Image Files (*.png; *.tif; *.jpg; *.jpeg; *.bmp)"
    fname, _ = fileDlg.getOpenFileName(self, 'Select Main Image', '', fileFilter)

    if len(fname) > 0:
      newIm = np.array(Image.open(fname))
      self.resetMainImg(newIm)

  @Slot()
  @applyWaitCursor
  def estBoundsBtnClicked(self):
    self.compMgr.rmComps()
    compVertices = getVertsFromBwComps(getBwComps(self.mainImgItem.image))
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
    self.compImg.setClickable(self.allowEditsChk.isChecked())

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
    vertBox = getClippedBbox(self.mainImgItem.image.shape, vertBox, sideLen)
    # Create square from bounding box
    compVerts = []
    compVerts.append([vertBox[0,0], vertBox[0,1]])
    compVerts.append([vertBox[1,0], vertBox[0,1]])
    compVerts.append([vertBox[1,0], vertBox[1,1]])
    compVerts.append([vertBox[0,0], vertBox[1,1]])
    compVerts = np.vstack(compVerts)

    newComp = makeCompDf()
    newComp[TC.VERTICES.name] = [compVerts]
    self.compMgr.addComps(newComp)


  @Slot(object)
  @applyWaitCursor
  def updateCurComp(self, newComp: df):
    mainImg = self.mainImgItem.image
    margin = self.regCtrlEditor[RCEV.MARGIN].value()
    segThresh = self.regCtrlEditor[RCEV.SEG_THRESH].value()
    prevComp = self.compImg.comp
    rmPrevComp = self.compImg.updateAll(mainImg, newComp, margin, segThresh)
    # If all old vertices were deleted AND we switched images, signal deletion
    # for the previous focused component
    if rmPrevComp:
      self.compMgr.rmComps(prevComp.instanceId)

    self.curCompIdLbl.setText(f'Component ID: {newComp[TC.INST_ID.name].squeeze()}')

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
  app = pg.mkQApp()
  win = MainWindow()
  win.show()
  app.exec()