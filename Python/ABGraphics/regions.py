import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

import numpy as np
import cv2 as cv

from typing import Tuple

from SchemeEditor import SchemeEditor
from constants import SchemeValues as SV

class VertexRegion(pg.ImageItem):
  scheme = None

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.offset = [0,0]

  def updateVertices(self, newVerts):
    if newVerts.shape[0] == 0:
      # No need to look for polygons if vertices are empty
      self.setImage(np.zeros((1,1), dtype='uint8'))
      return
    self.offset = newVerts.min(0)
    newVerts -= self.offset
    newImgShape = (newVerts.max(0)+1)[::-1]
    regionData = np.zeros(newImgShape, dtype='uint8')
    cv.fillPoly(regionData, [newVerts], 1)
    # Make vertices full brightness
    regionData[newVerts[:,1], newVerts[:,0]] = 2
    self.setImage(regionData, levels=[0,2], lut=self.getLUTFromScheme())
    self.setPos(*self.offset)

  def embedMaskInImg(self, toEmbedShape: Tuple[int, int]):
    outImg = np.zeros(toEmbedShape, dtype=bool)
    selfShape = self.image.shape
    # Offset is x-y, shape is row-col. So, swap order of offset relative to current axis
    embedSlices = [slice(self.offset[1-ii], selfShape[ii]+self.offset[1-ii]) for ii in range(2)]
    outImg[embedSlices[0], embedSlices[1]] = self.image
    return outImg

  @staticmethod
  def getLUTFromScheme():
    if VertexRegion.scheme is None:
      VertexRegion.scheme = SchemeEditor()
    fillClr, vertClr = VertexRegion.scheme.getFocImgProps((SV.foc_fillColor, SV.foc_vertColor))
    lut = [(0,0,0,0)]
    for clr in fillClr, vertClr:
      lut.append(clr.getRgb())
    return np.array(lut, dtype='uint8')

  @staticmethod
  def setScheme(scheme: SchemeEditor):
    VertexRegion.scheme = scheme

class SaveablePolyROI(pg.PolyLineROI):
  def __init__(self, *args, **kwargs):
    # Since this won't execute until after module import, it doesn't cause
    # a dependency
    super().__init__(*args, **kwargs)
    # Force new menu options
    self.getMenu()

  def getMenu(self, *args, **kwargs):
    '''
    Adds context menu option to add current ROI area to existing region
    '''
    if self.menu is None:
      menu = super().getMenu()
      addAct = QtGui.QAction("Add to Region", menu)
      menu.addAction(addAct)
      self.addAct = addAct
      self.menu = menu
    return self.menu

  def getImgMask(self, imgItem: pg.ImageItem):
    imgMask = np.zeros(imgItem.image.shape[0:2], dtype='bool')
    roiSlices,_ = self.getArraySlice(imgMask, imgItem)
    # TODO: Clip regions that extend beyond image dimensions
    roiSz = [curslice.stop - curslice.start for curslice in roiSlices]
    # renderShapeMask takes width, height args. roiSlices has row/col sizes,
    # so switch this order when passing to renderShapeMask
    roiSz = roiSz[::-1]
    roiMask = self.renderShapeMask(*roiSz).astype('uint8')
    # Also, the return value for renderShapeMask is given in col-major form.
    # Transpose this, since all other data is in row-major.
    roiMask = roiMask.T
    imgMask[roiSlices[0], roiSlices[1]] = roiMask
    return imgMask