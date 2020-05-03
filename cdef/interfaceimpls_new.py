from typing import Tuple

from pathlib import Path

import imageprocessing.algorithms as alg
from imageprocessing.common import Image
from imageprocessing.processing import ImageIO, ProcessStage, AtomicFunction, Process
from pyqtgraph.parametertree import Parameter
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

import cv2 as cv
import numpy as np

from cdef.frgraphics.parameditors import FR_SINGLETON
from cdef.structures import FRParam


class FREditableProcessIO(ImageIO):
  def __init__(self, ownerObj, **kwargs):
    self.ownerObj = ownerObj
    super().__init__(**kwargs)

  def __getitem__(self, key):
    retVal = super().__getitem__(key)
    if isinstance(retVal, Parameter):
      return retVal.value()
    else:
      return retVal

# Shorthand for convenience
register = FR_SINGLETON.algParamMgr.registerProp
def unpackStages(algName: str, stage: ProcessStage, paramParent: Tuple[str]=None):
  curGroup = FRParam(name=stage.name, valType='group', value=[])
  register(algName, curGroup, paramParent)
  if paramParent is None:
    paramParent = ()
  paramParent += stage.name,
  if isinstance(stage, AtomicFunction):
    pass
  else:
    # else: # Process
    stage: Process
    procInpt = FREditableProcessIO(stage, **stage.input)
    stage.input = procInpt
    for param in procInpt.parameterKeys:
      val = procInpt[param]
      curParam = FRParam(name=param, value=val)
      procInpt[param] = register(algName, curParam, paramParent, asProperty=False)
    for childStage in stage.stages:
      unpackStages(algName, childStage, paramParent=paramParent)
  return curGroup

if __name__ == '__main__':
  img = Image(Path('C:/Users/ntjes/Desktop/Git/cdef/images/circuitBoard.png'))
  app = pg.mkQApp()
  # res = ComponentDetection.findComponentsOtsu(img)
  # io = ImageIO(image=np.array([[[1,1,1]]], dtype='uint8'))
  io = ImageIO(image=img)
  process = io.initProcess("Find Components Otsu")
  p = alg.otsuThresholdProcess(binaryOut=True, smoothingKernelSize=11)
  process.addProcess(p)
  process.addProcess(alg.morphologyExProcess(cv.MORPH_OPEN, ksize=5))
  process.addProcess(alg.morphologyExProcess(cv.MORPH_CLOSE, ksize=7))
  algName = process.name
  FR_SINGLETON.algParamMgr.registerClass(FRParam(algName), overrideName=algName)()
  outParam = unpackStages(algName, process)
  mgr = FR_SINGLETON.algParamMgr
  pushbtn = QtWidgets.QPushButton('Run')
  def doRun():
    process.run(force=True)
    process.plotStages()
  pushbtn.clicked.connect(doRun)
  mgr.layout().addWidget(pushbtn)
  FR_SINGLETON.algParamMgr.showMaximized()
  app.exec()