# -*- coding: utf-8 -*-
"""
Simple example of loading UI template created with Qt Designer.

This example uses uic.loadUiType to parse and load the ui at runtime. It is also
possible to pre-compile the .ui file using pyuic (see VideoSpeedTest and
ScatterPlotSpeedTest examples; these .ui files have been compiled with the
tools/rebuildUi.py script).
"""
import PySide2
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
QInputDialog = QtWidgets.QInputDialog
QSettings = QtCore.QSettings
QIntValidator = QtGui.QIntValidator
QDoubleValidator = QtGui.QDoubleValidator
Signal = QtCore.pyqtSignal
Slot = QtCore.pyqtSlot

from inspect import getmembers

import pickle as pkl

import numpy as np
from PIL import Image
import cv2 as cv

from processing import getComps, segmentComp
from graphicshelpers import applyWaitCursor

import os
from glob import glob
from functools import partial
from component import Component, ComponentMgr

## Define main window class from template
path = os.path.dirname(os.path.abspath(__file__))
uiFile = os.path.join(path, 'imgAnnotator.ui')
WindowTemplate, TemplateBaseClass = pg.Qt.loadUiType(uiFile)

class MainWindow(TemplateBaseClass):
  # Alerts GUI that a layout (either new or overwriting old) was saved
  sigLayoutSaved = Signal()
  
  def __init__(self):
    # Configure pg to correctly read image dimensions
    pg.setConfigOption(imageAxisOrder='row-major')

    TemplateBaseClass.__init__(self)
    #self.setWindowTitle('pyqtgraph example: Qt Designer')

    # Create the main window
    self.ui = WindowTemplate()
    self.ui.setupUi(self)

    # Flesh out pg components
    # ---------------
    # MAIN IMAGE
    # ---------------
    item = pg.ImageItem(np.array(Image.open('../fast.tif')), axisOrder='row-major')
    # Ensure image will remain in background of window
    item.setZValue(-100)
    self.ui.mainImg.addItem(item)
    self.ui.mainImg.setAspectLocked(True)
    self.mainImgItem = item

    # ---------------
    # COMPONENT IMAGE
    # ---------------
    item = pg.ImageItem(np.array(0.).reshape((1,1)))
    self.ui.compImg.addItem(item)
    self.ui.compImg.setAspectLocked(True)
    self.compImgItem = item

    # ---------------
    # INPUT VALIDATORS
    # ---------------
    intVdtr = QIntValidator()
    floatVdtr = QDoubleValidator()
    self.ui.marginEdit.setValidator(intVdtr)
    self.ui.segThreshEdit.setValidator(floatVdtr)
    self.ui.seedThreshEdit.setValidator(floatVdtr)

    # ---------------
    # COMPONENT MANAGER
    # ---------------
    self.compMgr = ComponentMgr(self.ui.mainImg)
    self.compMgr.sigCompClicked.connect(self.updateCurComp)
    
    # ---------------
    # LOAD LAYOUT OPTIONS
    # ---------------
    self.populateLoadLayoutOptions()
    # Start with docks in default position
    self.loadLayoutActionTriggered('Default')


    # ---------------
    # UI ELEMENT SIGNALS
    # ---------------
    # Buttons
    self.ui.newImgBtn.clicked.connect(self.newImgBtnClicked)
    self.ui.estBoundsBtn.clicked.connect(self.estBoundsBtnClicked)
    self.ui.clearBoundsBtn.clicked.connect(self.clearBoudnsBtnClicked)
    
    # Menu options
    self.ui.saveLayout.triggered.connect(self.saveLayoutActionTriggered)
    self.sigLayoutSaved.connect(self.populateLoadLayoutOptions)
    
  def populateLoadLayoutOptions(self):
    layoutMenu = self.ui.loadLayout
    # Remove existing menus so only the current file system setup is in place
    for action in layoutMenu.children():
      layoutMenu.removeAction(action)
    layouts = glob('./Layouts/*.dockstate')
    for layout in layouts:
      # glob returns entire filepath, so keep only filename as layout name
      name = os.path.basename(layout)
      # Also strip file extension
      name = name[0:name.rfind('.')]
      curAction = layoutMenu.addAction(name)
      curAction.triggered.connect(partial(self.loadLayoutActionTriggered, name))
    

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
    sampleComps = getComps(self.mainImgItem.image)
    contours, _ = cv.findContours(sampleComps.astype('uint8'), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    components = []
    for contour in contours:
      newComp = Component()
      newComp.vertices = contour[:,0,:]
      components.append(newComp)
    self.compMgr.addComps(components)

  @Slot()
  @applyWaitCursor
  def clearBoudnsBtnClicked(self):
    self.compMgr.rmComps()
    
  @Slot()
  def loadLayoutActionTriggered(self, layoutName):
    with open(f'./Layouts/{layoutName}.dockstate', 'rb') as savedSettings:
      dockStates = pkl.load(savedSettings)
    self.restoreState(dockStates)
  
  @Slot()
  def saveLayoutActionTriggered(self):
    dockStates = self.saveState()
    saveName, ok = QtWidgets.QInputDialog() \
    .getText(self, 'Layout Name', 'Layout Name:', QtWidgets.QLineEdit.Normal)
    if ok:
      # Prevent overwriting default layout
      if saveName.lower() == 'default':
        QtGui.QMessageBox().information(self, 'Error During Save',
                    'Cannot overwrite default layout.', QtGui.QMessageBox.Ok)
        return
      with open(f'./Layouts/{saveName}.dockstate', 'wb') as saveFile:
        pkl.dump(dockStates, saveFile)
    self.sigLayoutSaved.emit()
      
  @applyWaitCursor
  def updateCurComp(self, newComp: Component):
    mainImg = self.mainImgItem.image
    margin = int(self.ui.marginEdit.text())

    bbox = np.vstack((newComp.vertices.min(0),
          newComp.vertices.max(0)))
    # Account for margins
    for ii in range(2):
      bbox[0,ii] = np.maximum(0, bbox[0,ii]-margin)
      bbox[1,ii] = np.minimum(mainImg.shape[1-ii], bbox[1,ii]+margin)

    newCompImg = mainImg[bbox[0,1]:bbox[1,1], bbox[0,0]:bbox[1,0],:]
    segImg = segmentComp(newCompImg, float(self.ui.segThreshEdit.text()))
    self.compImgItem.setImage(segImg)

  def resetMainImg(self, newIm: np.array):
    self.mainImgItem.setImage(newIm)
    self.compMgr.rmComps('all')


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
  app = pg.mkQApp()
  win = MainWindow()
  win.show()
  app.exec()