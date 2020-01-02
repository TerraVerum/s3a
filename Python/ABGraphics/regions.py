import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

import numpy as np
import cv2 as cv

from typing import Tuple

from SchemeEditor import SchemeEditor
from constants import SchemeValues as SV

class VertexRegion(pg.ImageItem):
  scheme = SchemeEditor()

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.offset = np.array([0,0], dtype=np.int)

  def updateVertices(self, newVerts):
    # If only one vertex list is passed, wrap it
    if isinstance(newVerts, np.ndarray):
      newVerts = [newVerts]
    if len(newVerts) == 0:
      self.setImage(np.zeros((1,1), dtype='bool'))
      return
    allVerts: np.ndarray = np.vstack(newVerts)
    self.offset = allVerts.min(0)
    for vertList in newVerts:
      vertList -= self.offset
    allVerts -= self.offset

    newImgShape = (allVerts.max(0)+1)[::-1]
    regionData = np.zeros(newImgShape, dtype='uint8')
    cv.fillPoly(regionData, newVerts, 1)
    # Make vertices full brightness
    regionData[allVerts[:,1], allVerts[:,0]] = 2
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

class MultiRegionPlot(pg.PlotDataItem):
  scheme = SchemeEditor()

  def __init__(self, *args, **kargs):
    super().__init__(*args, **kargs, connect='finite')
    self.regions = []
    self.ids = []
    self._nanSep = np.empty((1,2))
    self._nanSep.fill(np.nan)

  def resetRegionList(self, newIds=[], newRegions = []):
    self.regions = newRegions
    self.ids = newIds
    self.updatePlot()

  def setRegions(self, regionIds, vertices):
    '''
    If the region already exists, update it. Otherwise, append to the list.
    If region vertices are empty, remove the region
    '''
    # Wrap single region instances in list to allow batch processing
    if isinstance(regionIds, int):
      regionIds = [regionIds]
      vertices = [vertices]
    for curId, curVerts in zip(regionIds, vertices):
      try:
        regionIdx = self.ids.index(curId)
        if len(curVerts) == 0:
          del self.regions[regionIdx]
          del self.ids[regionIdx]
        else:
          # Add nan values to indicate separate regions once all verts
          # are concatenated for plotting
          curVerts = np.vstack((curVerts, self._nanSep))
          self.regions[regionIdx] = curVerts
      except ValueError:
        if len(curVerts) > 0:
          self.ids.append(curId)
          curVerts = np.vstack((curVerts, self._nanSep))
          self.regions.append(curVerts)
    self.updatePlot()

  def updatePlot(self):
    # -----------
    # Update data
    # -----------
    concatData = [], []
    if len(self.regions) > 0:
      # We have regions to plot
      concatData = np.vstack(self.regions)
      concatData = (concatData[:,0], concatData[:,1])

    # -----------
    # Update scheme
    # -----------
    boundClr, boundWidth = MultiRegionPlot.scheme.getCompProps(
                             (SV.boundaryColor, SV.boundaryWidth))
    pltPen = pg.mkPen(boundClr, width=boundWidth)
    self.setData(*concatData, pen=pltPen)

  @staticmethod
  def setScheme(scheme):
    MultiRegionPlot.scheme = scheme