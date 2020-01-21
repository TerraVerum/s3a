from typing import Tuple

import cv2 as cv
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

from .parameditors import SCHEME_HOLDER
from ..constants import SchemeValues as SV
from ..processing import splitListAtNans


class VertexRegion(pg.ImageItem):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self._offset = np.array([0,0], dtype=int)
    self.verts = [np.zeros((0,2), dtype=int)]

  def updateVertices(self, newVerts: np.ndarray):
    self.verts = newVerts.copy()

    if len(newVerts) == 0:
      self.setImage(np.zeros((1,1), dtype='bool'))
      return
    self._offset = np.nanmin(newVerts, 0)
    newVerts -= self._offset

    # cv.fillPoly requires list-of-lists format
    fillPolyArg = splitListAtNans(newVerts)
    newImgShape = (np.nanmax(newVerts, 0)+1)[::-1]
    regionData = np.zeros(newImgShape, dtype='uint8')
    cv.fillPoly(regionData, fillPolyArg, 1)
    # Make vertices full brightness
    nonNanVerts = newVerts[np.invert(np.isnan(newVerts[:,0])),:]
    regionData[nonNanVerts[:,1], nonNanVerts[:,0]] = 2
    self.setImage(regionData, levels=[0,2], lut=self.getLUTFromScheme())
    self.setPos(*self._offset)

  def embedMaskInImg(self, toEmbedShape: Tuple[int, int]):
    outImg = np.zeros(toEmbedShape, dtype=bool)
    selfShape = self.image.shape
    # Offset is x-y, shape is row-col. So, swap order of offset relative to current axis
    embedSlices = [slice(self._offset[1-ii], selfShape[ii]+self._offset[1-ii]) for ii in range(2)]
    outImg[embedSlices[0], embedSlices[1]] = self.image
    return outImg

  @staticmethod
  def getLUTFromScheme():
    fillClr, vertClr = SCHEME_HOLDER.scheme.getFocImgProps((SV.REG_FILL_COLOR, SV.REG_VERT_COLOR))
    lut = [(0,0,0,0)]
    for clr in fillClr, vertClr:
      lut.append(clr.getRgb())
    return np.array(lut, dtype='uint8')

class SaveablePolyROI(pg.PolyLineROI):
  def __init__(self, initialPoints=None, *args, **kwargs):
    if initialPoints is None:
      initialPoints = []
    # Since this won't execute until after module import, it doesn't cause
    # a dependency
    super().__init__(initialPoints, *args, **kwargs)
    # Force new menu options
    self.finishPolyAct = QtGui.QAction()
    self.getMenu()

  def getMenu(self, *args, **kwargs):
    """
    Adds context menu option to add current ROI area to existing region
    """
    if self.menu is None:
      menu = super().getMenu()
      finishPolyAct = QtGui.QAction("Finish Polygon", menu)
      menu.addAction(finishPolyAct)
      self.finishPolyAct = finishPolyAct
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
  def __init__(self, *args, **kargs):
    super().__init__(*args, **kargs, connect='finite')
    self.regions = []
    self.ids = []
    self._nanSep = np.empty((1,2))
    self._nanSep.fill(np.nan)

  def resetRegionList(self, newIds=None, newRegions=None):
    if newRegions is None:
      newRegions = []
    if newIds is None:
      newIds = []
    self.regions = []
    self.ids = []
    self[newIds] = newRegions

  def updatePlot(self):
    # -----------
    # Update data
    # -----------
    concatData = [], []
    if len(self.regions) > 0:
      # Before stacking regions, add first point of region to end of region vertices.
      # This will make the whole region connected in the output plot
      # Insert nan to make separate components unconnected
      plotRegions = []
      for region in self.regions:
        region = np.vstack((region, region[0,:], self._nanSep))
        plotRegions.append(region)
      # We have regions to plot
      concatData = np.vstack(plotRegions)
      concatData = (concatData[:,0], concatData[:,1])

    # -----------
    # Update scheme
    # -----------
    boundClr, boundWidth = SCHEME_HOLDER.scheme.getCompProps(
                             (SV.BOUNDARY_COLOR, SV.BOUNDARY_WIDTH))
    pltPen = pg.mkPen(boundClr, width=boundWidth)
    self.setData(*concatData, pen=pltPen)

  def __getitem__(self, regionIds):
    """
    Allows retrieval of vertex list for a given id list
    """
    # Wrap single region instances in list to allow batch processing
    returnSingle = False
    if not hasattr(regionIds, '__iter__'):
      returnSingle = True
      regionIds = np.array([regionIds])
    outList = np.empty(regionIds.size, dtype=object)

    for ii, curId in enumerate(regionIds):
      try:
        regionIdx = self.ids.index(curId)
        # Found the region
        outList[ii] = self.regions[regionIdx]
      except ValueError:
        # Requested ID was not in the displayed regions. Indicate with '[]'
        outList[ii] = []
    # Unwrap single value at end
    if returnSingle:
      outList = outList[0]
    return outList

  def __setitem__(self, regionIds, newVerts):
    """
    If the region already exists, update it. Otherwise, append to the list.
    If region vertices are empty, remove the region
    """
    if not hasattr(regionIds, '__iter__'):
      regionIds = [regionIds]
      newVerts = [newVerts]
    elif len(newVerts) != len(regionIds):
      # Same value for all specified region ids
      newVerts = [newVerts for _ in regionIds]
    regionIds = np.array(regionIds)
    newVerts = np.array(newVerts)

    emptyVertIdxs = np.array([len(verts) == 0 for verts in newVerts], dtype=bool)
    # If new verts are empty, delete the region
    keepIds = regionIds[~emptyVertIdxs]
    vertsAtKeepIds = newVerts[~emptyVertIdxs]
    rmIds = regionIds[emptyVertIdxs]
    for curId, curVerts in zip(keepIds, vertsAtKeepIds):
      # Append if not already present in list
      try:
        idIdx = self.ids.index(curId)
        self.regions[idIdx] = curVerts
      except ValueError:
        self.ids.append(curId)
        self.regions.append(curVerts)
    for curId in rmIds:
      try:
        idIdx = self.ids.index(curId)
        del self.ids[idIdx]
        del self.regions[idIdx]
      except ValueError:
        # The Id was initialized to empty before it was actually plotted
        pass
    self.updatePlot()