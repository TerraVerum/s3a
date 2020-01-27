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
from Annotator.generalutils import splitListAtNans, coerceDfTypes

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
    nonNanVerts = newVerts[np.invert(np.isnan(newVerts[:,0])),:].astype(int)
    newImgShape = (np.max(newVerts, 0)+1)[::-1]
    regionData = np.zeros(newImgShape, dtype='uint8')
    cv.fillPoly(regionData, fillPolyArg, 1)
    # Make vertices full brightness
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
    fillClr, vertClr = SCHEME_HOLDER.scheme[SV.FOC_IMG_PARAMS,
                                            (SV.REG_FILL_COLOR, SV.REG_VERT_COLOR)]

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

def makeMultiRegionDf(numRows=1, whichCols=None) -> df:
  df_list = []
  if whichCols is None:
    whichCols = (TC.INST_ID, TC.VERTICES, TC.VALIDATED)
  elif isinstance(whichCols, ABParam):
    whichCols = [whichCols]
  for _ in range(numRows):
    # Make sure to construct a separate component instance for
    # each row no objects have the same reference
    df_list.append([field.value for field in whichCols])
  outDf = df(df_list, columns=whichCols)
  # Ensure base type fields are properly typed
  coerceDfTypes(outDf, whichCols)

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
    self.idPlts = ClickableScatterItem(pen=None)
    self.idPlts.sigClicked.connect(self.scatPltClicked)
    self._nanSep = np.empty((1,2)) * np.nan
    self.data = makeMultiRegionDf(0)

  @Slot(object, object)
  def scatPltClicked(self, plot, points):
    # Only send click signal for one point in the list
    self.sigIdClicked.emit(points[-1].data())

  def resetRegionList(self, newIds: Optional[Sequence]=None, newRegionDf: Optional[df]=None):
    if newIds is None:
      newIds = []
    if newRegionDf is None:
      newRegionDf = makeMultiRegionDf(0)
    self.data = makeMultiRegionDf(0)
    self[newIds,newRegionDf.columns] = newRegionDf

  def selectById(self, selectedIds):
    selectedClr = SCHEME_HOLDER.scheme[SV.COMP_PARAMS, SV.SELECTED_ID_BORDER]
    pens = np.empty(len(self.data), dtype=object)
    pens.fill(None)
    selectedIdxs = np.isin(self.data.index, selectedIds)
    pens[selectedIdxs] = pg.mkPen(selectedClr, width=3)
    self.idPlts.setPen(pens)

  def updatePlot(self):
    # -----------
    # Update scheme
    # -----------
    neededParams = (SV.VALID_ID_COLOR, SV.NONVALID_ID_COLOR,
                    SV.BOUNDARY_COLOR, SV.BOUNDARY_WIDTH,
                    SV.ID_FONT_SIZE, SV.SELECTED_ID_BORDER)
    validFill, nonValidFill, boundClr, boundWidth, idSz, selectedClr = \
      SCHEME_HOLDER.scheme[SV.COMP_PARAMS, neededParams]

    # -----------
    # Update data
    # -----------
    plotRegions = [np.ones((0,2))]
    idLocs = [np.ones((0,2))]

    for region in self.data.loc[:, TC.VERTICES]:
      idLoc = np.nanmean(region, 0).reshape(1,2)
      idLocs.append(idLoc)
      # Before stacking regions, add first point of region to end of region vertices.
      # This will make the whole region connected in the output plot
      # Insert nan to make separate components unconnected
      region = np.vstack((region, region[0,:], self._nanSep))
      plotRegions.append(region)
    idLocs = np.vstack(idLocs)
    # TODO: If the 'development' branch of pyqtgraph is set up, the clickable portion of each
    # plot will be the ID of the component. Otherwise it must be a non-descript item.
    #scatSymbols = [_makeTxtSymbol(str(curId), idSz) for curId in self.data.index]
    scatSymbols = [None for curId in self.data.index]

    brushes = np.empty(len(self.data), dtype=object)
    brushes.fill(pg.mkBrush(nonValidFill))
    brushes[self.data.loc[:, TC.VALIDATED]] = pg.mkBrush(validFill)

    self.idPlts.setData(x=idLocs[:,0], y=idLocs[:,1], size=idSz, brush=brushes,
                        data=self.data.index, symbol=scatSymbols)
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
    keysDf = makeMultiRegionDf(len(regionIds), setVals)
    keysDf = keysDf.set_index(regionIds)
    # Since we are only resetting one parameter (either valid or regions),
    # Make sure to keep the old parameter value for the unset index
    keysDf.update(self.data)
    keysDf.loc[regionIds, setVals] = vals
    self.data.update(keysDf)

    # Now we can add entries that weren't in our original dataframe
    # If not all set values were provided in the new dataframe, fix this by embedding
    # it into the default dataframe
    newDataDf = makeMultiRegionDf(np.sum(newEntryIdxs))
    newDataDf.loc[:, keysDf.columns] = keysDf.loc[newEntryIdxs, :]
    self.data = pd.concat((self.data, newDataDf))
    # Retain type information
    coerceDfTypes(self.data, makeMultiRegionDf(0).columns)
    self.updatePlot()

  def drop(self, ids):
    self.data.drop(index=ids, inplace=True)