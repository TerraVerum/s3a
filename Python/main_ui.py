# -*- coding: utf-8 -*-

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui, uic
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

from processing import getComps
from graphicshelpers import applyWaitCursor

import os
from glob import glob
from functools import partial
from component import Component, ComponentMgr

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
    item = pg.ImageItem(imgArray)
    # Ensure image will remain in background of window
    item.setZValue(-100)
    self.mainImg.addItem(item)
    self.mainImg.setAspectLocked(True)
    self.mainImgItem = item

    # ---------------
    # INPUT VALIDATORS
    # ---------------
    intVdtr = QIntValidator()
    floatVdtr = QDoubleValidator()
    self.marginEdit.setValidator(intVdtr)
    self.segThreshEdit.setValidator(floatVdtr)
    self.seedThreshEdit.setValidator(floatVdtr)

    # ---------------
    # COMPONENT MANAGER
    # ---------------
    self.compMgr = ComponentMgr(self.mainImg)
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
    self.newImgBtn.clicked.connect(self.newImgBtnClicked)
    self.estBoundsBtn.clicked.connect(self.estBoundsBtnClicked)
    self.clearBoundsBtn.clicked.connect(self.clearBoudnsBtnClicked)

    # Menu options
    self.saveLayout.triggered.connect(self.saveLayoutActionTriggered)
    self.sigLayoutSaved.connect(self.populateLoadLayoutOptions)

  def populateLoadLayoutOptions(self):
    layoutMenu = self.loadLayout
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

  @Slot(object)
  @applyWaitCursor
  def updateCurComp(self, newComp: Component):
    mainImg = self.mainImgItem.image
    margin = int(self.marginEdit.text())
    segThresh = float(self.segThreshEdit.text())

    self.compImg.update(mainImg, newComp, margin, segThresh)

  def resetMainImg(self, newIm: np.array):
    self.mainImgItem.setImage(newIm)
    self.compMgr.rmComps('all')


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
  app = pg.mkQApp()
  win = MainWindow()
  win.show()
  app.exec()