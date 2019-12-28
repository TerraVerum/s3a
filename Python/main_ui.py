# -*- coding: utf-8 -*-
"""
Simple example of loading UI template created with Qt Designer.

This example uses uic.loadUiType to parse and load the ui at runtime. It is also
possible to pre-compile the .ui file using pyuic (see VideoSpeedTest and
ScatterPlotSpeedTest examples; these .ui files have been compiled with the
tools/rebuildUi.py script).
"""

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
QInputDialog = QtWidgets.QInputDialog
QSettings = QtCore.QSettings
Signal = QtCore.pyqtSignal
Slot = QtCore.pyqtSlot

import numpy as np
from PIL import Image
from skimage.morphology import dilation
from skimage.segmentation import quickshift
import cv2 as cv

import os
from component import Component, ComponentMgr

## Define main window class from template
path = os.path.dirname(os.path.abspath(__file__))
uiFile = os.path.join(path, 'imgAnnotator.ui')
WindowTemplate, TemplateBaseClass = pg.Qt.loadUiType(uiFile)

class MainWindow(TemplateBaseClass):
  def __init__(self):
    # Configure pg to correctly read image dimensions
    pg.setConfigOption('imageAxisOrder', 'row-major')
    
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

    self.compImg = {};
    self.compImgItem = item

    # ---------------
    # COMPONENT MANAGER
    # ---------------
    self.compMgr = ComponentMgr(self.ui.mainImg)
    self.compMgr.sigCompClicked.connect(self.updateCurComp)

    # ---------------
    # UI ELEMENT SIGNALS
    # ---------------
    self.ui.newImgBtn.clicked.connect(self.newImgBtnClicked)
    self.ui.estBoundsBtn.clicked.connect(self.estBoundsBtnClicked)
    self.ui.clearBoundsBtn.clicked.connect(self.clearBoudnsBtnClicked)

  @Slot()
  def newImgBtnClicked(self):
    fileDlg = QtWidgets.QFileDialog()
    fileFilter = "Image Files (*.png; *.tif; *.jpg; *.jpeg; *.bmp)"
    fname, _ = fileDlg.getOpenFileName(self, 'Select Main Image', '', fileFilter)

    if len(fname) > 0:
      newIm = np.array(Image.open(fname))
      self.resetMainImg(newIm)
    
  @Slot()
  def estBoundsBtnClicked(self):
    sampleComps = np.zeros(self.mainImgItem.image.shape[0:2], dtype='bool')
    sampleComps[[100, 200], [75, 75]] = True
    sampleComps = dilation(sampleComps, np.ones((25, 75)))
    contours, _ = cv.findContours(sampleComps.astype('uint8'), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    components = []
    for contour in contours:
      newComp = Component()
      newComp.vertices = contour[:,0,:]
      components.append(newComp)
    self.compMgr.addComps(components)
    
  @Slot()
  def clearBoudnsBtnClicked(self):
    self.compMgr.rmComps()

  def updateCurComp(self, newComp: Component):
    mainImg = self.mainImgItem.image
    
    bbox = np.vstack((newComp.vertices.min(0),
          newComp.vertices.max(0)))
    newCompImg = mainImg[bbox[0,1]:bbox[1,1], bbox[0,0]:bbox[1,0],:]
    self.compImgItem.setImage(newCompImg)
    
    
  def resetMainImg(self, newIm: np.array):
    self.mainImg['item'].setImage(newIm)  

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
  app = pg.mkQApp()
  win = MainWindow()
  win.show()
  app.exec()