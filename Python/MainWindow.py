# -*- coding: utf-8 -*-

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui, uic
Slot = QtCore.pyqtSlot
Signal = QtCore.pyqtSignal

import numpy as np
from PIL import Image

from processing import getBwComps, getVertsFromBwComps
from ABGraphics.utils import applyWaitCursor, dialogSaveToFile, addDirItemsToMenu, attemptLoadSettings
from SchemeEditor import SchemeEditor
from component import Component, ComponentMgr
from constants import SCHEMES_DIR, LAYOUTS_DIR

from os.path import join
import os

# Configure pg to correctly read image dimensions
pg.setConfigOptions(imageAxisOrder='row-major')

class MainWindow(QtWidgets.QMainWindow):
  # Alerts GUI that a layout (either new or overwriting old) was saved
  sigLayoutSaved = Signal()

  def __init__(self):
    super().__init__()
    uiPath = os.path.dirname(os.path.abspath(__file__))
    uiFile = os.path.join(uiPath, 'imgAnnotator.ui')
    uic.loadUi(uiFile, self)

    # Flesh out pg components
    # ---------------
    # MAIN IMAGE
    # ---------------
    imgArray = np.array(Image.open('../fast.tif'))
    #imgArray = None
    item = pg.ImageItem(imgArray)
    # Ensure image will remain in background of window
    item.setZValue(-100)
    self.mainImg.addItem(item)
    self.mainImg.setAspectLocked(True)
    self.mainImgItem = item

    # ---------------
    # INPUT VALIDATORS
    # ---------------
    intVdtr = QtGui.QIntValidator()
    floatVdtr = QtGui.QDoubleValidator()
    self.marginEdit.setValidator(intVdtr)
    self.segThreshEdit.setValidator(floatVdtr)
    self.seedThreshEdit.setValidator(floatVdtr)

    # ---------------
    # LOAD LAYOUT OPTIONS
    # ---------------
    self.populateLoadLayoutOptions()
    # Start with docks in default position
    self.loadLayoutActionTriggered('Default')

    # ---------------
    # COMPONENT MANAGER
    # ---------------
    self.compMgr = ComponentMgr(self.mainImg, self.mainImgItem)
    self.compMgr.sigCompClicked.connect(self.updateCurComp)

    # ---------------
    # LOAD SCHEME OPTIONS
    # ---------------
    self.scheme = SchemeEditor()
    self.populateSchemeOptions()
    # Attach scheme to all UI children
    self.compImg.setScheme(self.scheme)
    Component.setScheme(self.scheme)
    ComponentMgr.setScheme(self.scheme)

    # ---------------
    # UI ELEMENT SIGNALS
    # ---------------
    # Buttons
    self.newImgBtn.clicked.connect(self.newImgBtnClicked)
    self.estBoundsBtn.clicked.connect(self.estBoundsBtnClicked)
    self.clearBoundsBtn.clicked.connect(self.clearBoundsBtnClicked)
    self.clearRegionBtn.clicked.connect(self.clearRegionBtnClicked)
    self.resetRegionBtn.clicked.connect(self.resetRegionBtnClicked)

    # Edit fields
    self.seedThreshEdit.editingFinished.connect(self.seedThreshChanged)
    # Note: This signal must be false-triggered on startup to propagate
    # the field's initial value
    self.seedThreshEdit.editingFinished.emit()

    # Menu options
    self.saveLayout.triggered.connect(self.saveLayoutActionTriggered)
    self.sigLayoutSaved.connect(self.populateLoadLayoutOptions)

    self.newScheme.triggered.connect(self.scheme.show)

    # Scheme editor
    self.scheme.sigSchemeSaved.connect(self.populateSchemeOptions)
    # When a new scheme is created, switch to that scheme
    self.scheme.sigSchemeSaved.connect(self.loadSchemeActionTriggered)

  def populateLoadLayoutOptions(self):
    layoutGlob = join(LAYOUTS_DIR, '*.dockstate')
    addDirItemsToMenu(self.loadLayout, layoutGlob, self.loadLayoutActionTriggered)

  @Slot(str)
  def loadLayoutActionTriggered(self, layoutName):
    layoutFilename = join(LAYOUTS_DIR, f'{layoutName}.dockstate')
    dockStates = attemptLoadSettings(layoutFilename)
    if dockStates is not None:
      self.restoreState(dockStates)

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

  @Slot()
  def clearRegionBtnClicked(self):
    # Reset drawn comp vertices to nothing
    # Only perform action if image currently exists
    if self.compImg.compImgItem.image is None:
      return
    self.compImg.updateRegion([])

  @Slot()
  def resetRegionBtnClicked(self):
    # Reset drawn comp vertices to nothing
    # Only perform action if image currently exists
    if self.compImg.compImgItem.image is None:
      return
    self.compImg.updateRegion(self.compImg.comp.vertices)

  @Slot()
  def seedThreshChanged(self):
    self.compImg.seedThresh = np.float(self.seedThreshEdit.text())

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
    components = []
    for verts in compVertices:
      newComp = Component()
      newComp.vertices = verts
      components.append(newComp)
    self.compMgr.addComps(components)

  @Slot()
  @applyWaitCursor
  def clearBoundsBtnClicked(self):
    self.compMgr.rmComps()

  @Slot()
  def saveLayoutActionTriggered(self):
    dockStates = self.saveState()
    dialogSaveToFile(self, dockStates, 'Layout Name', LAYOUTS_DIR, 'dockstate')
    self.sigLayoutSaved.emit()

  @Slot(object)
  @applyWaitCursor
  def updateCurComp(self, newComp: Component):
    mainImg = self.mainImgItem.image
    margin = int(self.marginEdit.text())
    segThresh = float(self.segThreshEdit.text())

    self.compImg.updateAll(mainImg, newComp, margin, segThresh)

  def resetMainImg(self, newIm: np.array):
    self.mainImgItem.setImage(newIm)
    self.compMgr.rmComps('all')


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
  app = pg.mkQApp()
  win = MainWindow()
  win.show()
  app.exec()