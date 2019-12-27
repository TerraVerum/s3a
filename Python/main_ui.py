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
import numpy as np
from PIL import Image

from skimage.morphology import dilation
import cv2 as cv

import os
from component import Component, ComponentMgr

## Define main window class from template
path = os.path.dirname(os.path.abspath(__file__))
uiFile = os.path.join(path, 'imgAnnotator.ui')
WindowTemplate, TemplateBaseClass = pg.Qt.loadUiType(uiFile)

class MainWindow(TemplateBaseClass):
  def __init__(self):
    TemplateBaseClass.__init__(self)
    #self.setWindowTitle('pyqtgraph example: Qt Designer')

    # Create the main window
    self.ui = WindowTemplate()
    self.ui.setupUi(self)

    # Flesh out pg components
    # ---------------
    # MAIN IMAGE
    # ---------------
    mainView = pg.ViewBox(lockAspect=True)
    item = pg.ImageItem(np.array(Image.open('../fast.tif')))
    mainView.addItem(item)
    self.ui.mainImg.setCentralItem(mainView)

    self.mainImg = {};
    self.mainImg['view'] = mainView
    self.mainImg['item'] = item

    # ---------------
    # COMPONENT IMAGE
    # ---------------
    compView = pg.ViewBox(lockAspect=True)
    item = pg.ImageItem(np.array(0.).reshape((1,1)))
    compView.addItem(item)
    self.ui.compImg.setCentralItem(compView)

    self.compImg = {};
    self.compImg['item'] = item
    self.compImg['view'] = compView

    # ---------------
    # COMPONENT MANAGER
    # ---------------
    self.compMgr = ComponentMgr(mainView)

    # ---------------
    # SIGNALS
    # ---------------
    self.ui.newImgBtn.clicked.connect(self.newImgBtnClicked)
    self.ui.estBoundsBtn.clicked.connect(self.estBounds)

  def newImgBtnClicked(self):
    fileDlg = QtWidgets.QFileDialog()
    fileFilter = "Image Files (*.png; *.tif; *.jpg; *.jpeg; *.bmp)"
    fname, _ = fileDlg.getOpenFileName(self, 'Select Main Image', '', fileFilter)

    if len(fname) > 0:
      newIm = np.array(Image.open(fname))
      self.resetMainImg(newIm)

  def resetMainImg(self, newIm: np.array):
    self.mainImg['item'].setImage(newIm)

  def estBounds(self):
    sampleComps = np.zeros(self.mainImg['item'].image.shape[0:2], dtype='bool')
    sampleComps[[100, 200], [75, 75]] = True
    sampleComps = dilation(sampleComps, np.ones((25, 25)))
    contours, _ = cv.findContours(sampleComps.astype('uint8'), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    components = []
    for contour in contours:
      newComp = Component()
      newComp.vertices = contour[:,0,:]
      components.append(newComp)
    self.compMgr.addComps(components)

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
  app = pg.mkQApp()
  win = MainWindow()
  win.show()
  app.exec()