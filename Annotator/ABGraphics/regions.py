from typing import Tuple, Sequence, Optional, Any

import cv2 as cv
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pandas import DataFrame as df
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

from Annotator.constants import TEMPLATE_COMP as TC
from Annotator.params import ABParamGroup, ABParam, newParam
from .parameditors import SCHEME_HOLDER
from .clickables import ClickableScatterItem
from ..constants import TEMPLATE_SCHEME_VALUES as SV
from ..processing import splitListAtNans

Signal = QtCore.pyqtSignal
Slot = QtCore.pyqtSlot

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

def _makeMultiRegionDf(numRows=1) -> df:
  df_list = []
  cols = [TC.VALIDATED, TC.VERTICES]
  for _ in range(numRows):
    # Make sure to construct a separate component instance for
    # each row no objects have the same reference
    df_list.append([field.value for field in cols])
  # Ensure 'valid' is boolean
  outDf = df(df_list, columns=cols)
  outDf[TC.VALIDATED] = outDf[TC.VALIDATED].astype('bool')
  return outDf

def _makeTxtSymbol(txt: str, fontSize: int):
  outSymbol = QtGui.QPainterPath()
  txtLabel = QtGui.QFont("Sans Serif", fontSize)
  txtLabel.setStyleStrategy(QtGui.QFont.PreferBitmap | QtGui.QFont.PreferQuality)
  outSymbol.addText(0, 0, txtLabel, txt)
  br = outSymbol.boundingRect()
  scale = min(1. / br.width(), 1. / br.height())
  tr = QtGui.QTransform()
  tr.scale(scale, scale)
  tr.translate(-br.x() - br.width()/2., -br.y() - br.height()/2.)
  outSymbol = tr.map(outSymbol)
  return outSymbol

class MultiRegionPlot(QtCore.QObject):
  sigIdClicked = Signal(int)
  # Helper class for IDE assistance during dataframe access
  def __init__(self, parent=None):
    super().__init__(parent)
    self.boundPlt = pg.PlotDataItem(connect='finite')
    self.validIdPlt = ClickableScatterItem(pen=None)
    self.nonValidIdPlt = ClickableScatterItem(pen=None)
    for plt in self.validIdPlt, self.nonValidIdPlt:
      plt.sigClicked.connect(self.scatPltClicked)
    self._nanSep = np.empty((1,2)) * np.nan
    self.data = _makeMultiRegionDf(0)

  @Slot(object, object)
  def scatPltClicked(self, plot, points):
    # Only send click signal for one point in the list
    self.sigIdClicked.emit(points[-1].data())

  def resetRegionList(self, newIds: Optional[Sequence]=None, vertValidDf: Optional[df]=None):
    if newIds is None:
      newIds = []
    if vertValidDf is None:
      vertValidDf = _makeMultiRegionDf(0)
    self.data = _makeMultiRegionDf(0)
    self[newIds,:] = vertValidDf

  def updatePlot(self):
    # -----------
    # Update scheme
    # -----------
    validFill, nonValidFill = SCHEME_HOLDER.scheme.getCompProps(
      [SV.VALID_ID_COLOR, SV.NONVALID_ID_COLOR]
    )
    boundClr, boundWidth, idSz = SCHEME_HOLDER.scheme.getCompProps(
      (SV.BOUNDARY_COLOR, SV.BOUNDARY_WIDTH, SV.ID_FONT_SIZE))

    # -----------
    # Update data
    # -----------
    validRegionIdxs = self.data.loc[:, TC.VALIDATED].to_numpy()
    plotRegions = [np.ones((0,2))]
    for regionIdxs, plt, pltFill in zip([validRegionIdxs, np.invert(validRegionIdxs)],
                                        [self.validIdPlt, self.nonValidIdPlt],
                                        [validFill, nonValidFill]):
      curRegionList = self.data.loc[regionIdxs,TC.VERTICES]
      curIdList = self.data.index[regionIdxs]
      idLocs = [np.ones((0,2))]
      for region in curRegionList:
        idLoc = np.nanmean(region, 0).reshape(1,2)
        idLocs.append(idLoc)
        # Before stacking regions, add first point of region to end of region vertices.
        # This will make the whole region connected in the output plot
        # Insert nan to make separate components unconnected
        region = np.vstack((region, region[0,:], self._nanSep))
        plotRegions.append(region)
      idLocs = np.vstack(idLocs)
      # Now that the list for valid or invalid plot centers is complete, place them in
      # the current plot
      # TODO: If the 'development' branch of pyqtgraph is set up, the clickable portion of each
      # plot will be the ID of the component. Otherwise it must be a non-descript item.
      #scatSymbols = [_makeTxtSymbol(str(curId), idSz) for curId in curIdList]
      scatSymbols = [None for curId in curIdList]
      plt.setData(x=idLocs[:,0], y=idLocs[:,1], size=idSz, brush=pltFill, data=curIdList, symbol=scatSymbols)

    # Finally finished createing region boundaries to plot
    plotRegions = np.vstack(plotRegions)
    boundPen = pg.mkPen(color=boundClr, width=boundWidth)
    self.boundPlt.setData(plotRegions[:,0], plotRegions[:,1], pen=boundPen)

  def __getitem__(self, keys: Tuple[Any,...]):
    """
    Allows retrieval of vertex/valid list for a given set of IDs
    """
    return self.data.loc[keys[0], keys[1:]]

  def __setitem__(self, keys: Tuple, vals: Sequence):
    if not isinstance(keys, tuple):
      # Only one key passed, assume ID
      regionIds = keys
      setVals = slice(None)
    elif len(keys) == 2:
      regionIds = keys[0]
      setVals = keys[1]
    else:
      regionIds = keys[0]
      setVals = keys[1:]
    # First update old entries
    newEntryIdxs = np.isin(regionIds, self.data.index, invert=True)
    keysDf = _makeMultiRegionDf(len(regionIds))
    keysDf = keysDf.set_index(regionIds)
    # Since we are only resetting one parameter (either valid or regions),
    # Make sure to keep the old parameter value for the unset index
    keysDf.update(self.data)
    keysDf.loc[regionIds, setVals] = vals
    self.data.update(keysDf)

    # Now we can add entries that weren't in our original dataframe
    self.data = pd.concat((self.data, keysDf.loc[newEntryIdxs,:]))
    # Make sure 'valid' is still bool after the operation
    self.data[TC.VALIDATED] = self.data[TC.VALIDATED].astype('bool')
    self.updatePlot()

  def drop(self, ids):
    self.data.drop(index=ids, inplace=True)