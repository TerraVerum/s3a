from __future__ import annotations

from typing import Tuple, Sequence, Optional, Any, List

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pandas import DataFrame as df
from pyqtgraph import arrayToQPath
from pyqtgraph.Qt import QtGui, QtCore
from skimage import color
from matplotlib import cm
from matplotlib.pyplot import colormaps

from s3a import FR_SINGLETON
from s3a.constants import REQD_TBL_FIELDS as RTF, FR_CONSTS
from s3a.generalutils import coerceDfTypes, stackedVertsPlusConnections, dynamicDocstring
from s3a.structures import FRParam, FRVertices, FRComplexVertices, OneDArr, BlackWhiteImg
from s3a.structures.typeoverloads import GrayImg
from . import imageareas
from .clickables import FRBoundScatterPlot

__all__ = ['FRMultiRegionPlot', 'FRVertexDefinedImg', 'FRRegionCopierPlot']

from ..models.editorbase import RunOpts

Signal = QtCore.Signal

def makeMultiRegionDf(numRows=1, whichCols=None, idList=None) -> df:
  df_list = []
  if whichCols is None:
    whichCols = (RTF.INST_ID, RTF.VERTICES, RTF.COMP_CLASS)
  elif isinstance(whichCols, FRParam):
    whichCols = [whichCols]
  for _ in range(numRows):
    # Make sure to construct a separate component instance for
    # each row no objects have the same reference
    df_list.append([field.value for field in whichCols])
  outDf = df(df_list, columns=whichCols)
  if idList is not None:
    outDf = outDf.set_index(idList)
  # Ensure base type fields are properly typed
  coerceDfTypes(outDf, whichCols)

  return outDf

def makeColorsDf(numRows=1, selected:np.ndarray=None, focused: np.ndarray=None,
                 compClasses: np.ndarray=None, idList: np.ndarray=None, convertClasses=True):
  """
  Helper for creating new dataframe holding information determining color data.
  `selected` and `focused` must be boolean arrays indicating whether or not each component
  is selected or focused, respectively.
  If `convertClasses` is `True`, converts string list of component classes into
  their integer index into `tableData`.
  """
  cols = ['selected', 'focused', 'compClass']
  outDf = pd.DataFrame(columns=cols)
  if selected is None:
    selected = np.zeros(numRows, bool)
  outDf['selected'] = selected
  if focused is None:
    focused = np.zeros(numRows, bool)
  outDf['focused'] = focused
  if compClasses is None:
    compClasses = np.zeros(numRows, int)
  elif convertClasses:
    compClasses = compClassToIndex(compClasses)
  outDf['compClass'] = compClasses
  if idList is not None:
    outDf = outDf.set_index(idList)
  return outDf

def compClassToIndex(compClasses: Sequence[str]):
  """
  Converts class names into integer indices
  :param compClasses: List of string class names
  :return: List of indices where each index corresponds to the class index from
    `tableData`.
  """
  return np.fromiter(map(FR_SINGLETON.tableData.compClasses.index, compClasses),
              dtype=int, count=len(compClasses))

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

def _makeBoundSymbol(verts: FRVertices):
  verts = verts - verts.min(0, keepdims=True)
  path = arrayToQPath(*verts.T, connect='finite')
  return path

@FR_SINGLETON.registerGroup(FR_CONSTS.CLS_MULT_REG_PLT)
class FRMultiRegionPlot(FRBoundScatterPlot):
  def __init__(self, parent=None):
    super().__init__(size=1, pxMode=False)
    FR_SINGLETON.colorScheme.registerFunc(self.resetColors, FR_CONSTS.CLS_MULT_REG_PLT.name,
                                          runOpts=RunOpts.ON_CHANGED)
    self.setParent(parent)
    self.setZValue(50)
    self.regionData = makeMultiRegionDf(0)
    self.colorData = makeColorsDf(0)
    self.cmap = np.array([])
    self.resetColors()

    # 'pointsAt' is an expensive operation if many points are in the scatterplot. Since
    # this will be called anyway when a selection box is made in the main image, disable
    # mouse click listener to avoid doing all that work for nothing.
    # self.centroidPlts.mouseClickEvent = lambda ev: None
    self.mouseClickEvent = lambda ev: None
    # Also disable sigClicked. This way, users who try connecting to this signal won't get
    # code that runs but never triggers
    # self.centroidPlts.sigClicked = None
    self.sigPointsClicked = None

  def resetRegionList(self, newIds: Optional[Sequence]=None, newRegionDf: Optional[df]=None):
    if newIds is None:
      newIds = []
    if newRegionDf is None:
      newRegionDf = makeMultiRegionDf(0)
    numRows = len(newRegionDf)
    self.regionData = makeMultiRegionDf(0)
    self.regionData = self.regionData.append(newRegionDf)
    if newIds is not None:
      self.regionData = self.regionData.set_index(newIds)
    self.colorData = makeColorsDf(numRows, compClasses=self.regionData[RTF.COMP_CLASS],
                                  idList=self.regionData.index, convertClasses=True)
    self.updatePlot()

  def selectById(self, selectedIds: OneDArr):
    """
    Marks 'selectedIds' as currently selected by changing their scheme to user-specified
    selection values.
    """
    self.updateSelected_focused(selectedIds=selectedIds)

  def focusById(self, focusedIds: OneDArr):
    """
    Colors 'focusedIds' to indicate they are present in a focused views.
    """
    self.updateSelected_focused(focusedIds=focusedIds)

  def updateSelected_focused(self, selectedIds:np.ndarray=None,
                             focusedIds: np.ndarray=None):
    """
    :param selectedIds: All currently selected Ids
    :param focusedIds: All currently focused Ids
    """
    if len(self.regionData) == 0:
      return
    for col, idList in zip(self.colorData.columns, [selectedIds, focusedIds]):
      if idList is None: continue
      self.colorData[col] = False
      self.colorData.loc[idList, col] = True
    self.resetColors()

  def updatePlot(self):
    # -----------
    # Update data
    # -----------
    boundLocs = []
    boundSymbs = []
    if self.regionData.empty:
      self.setData(x=[], y=[], data=[])
      return

    for region, _id in zip(self.regionData.loc[:, RTF.VERTICES],
                           self.regionData.index):
      concatRegion, isfinite = stackedVertsPlusConnections(region)
      boundLoc = np.nanmin(concatRegion, 0, keepdims=True)
      boundSymbol = pg.arrayToQPath(*(concatRegion-boundLoc).T, connect=isfinite)

      boundLocs.append(boundLoc)
      boundSymbs.append(boundSymbol)

    plotRegions = np.vstack(boundLocs)
    self.setData(*plotRegions.T, symbol=boundSymbs,
                          data=self.regionData.index)
    self.resetColors()

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
    newEntryIdxs = np.isin(regionIds, self.regionData.index, invert=True)
    keysDf = makeMultiRegionDf(len(regionIds), setVals)
    keysDf = keysDf.set_index(regionIds)
    # Since we may only be resetting one parameter (either valid or regions),
    # Make sure to keep the old parameter value for the unset index
    keysDf.update(self.regionData)
    keysDf.loc[regionIds, setVals] = vals
    self.regionData.update(keysDf)

    # Now we can add entries that weren't in our original dataframe
    # If not all set values were provided in the new dataframe, fix this by embedding
    # it into the default dataframe
    newDataDf = makeMultiRegionDf(int(np.sum(newEntryIdxs)), idList=regionIds[newEntryIdxs])
    newDataDf.loc[:, keysDf.columns] = keysDf.loc[newEntryIdxs, :]
    self.regionData = pd.concat((self.regionData, newDataDf))
    # Retain type information
    coerceDfTypes(self.regionData, makeMultiRegionDf(0).columns)
    self.updatePlot()

  @dynamicDocstring(cmapVals=colormaps())
  def resetColors(self, penWidth=0, penColor='w', selectedFill='00f', focusedFill='f00', classColormap='tab10',
                  fillAlpha=0.7):
    """
    Assigns colors from the specified colormap to each unique class
    :param penWidth: Width of the pen in pixels
    :param penColor:
      helpText: Color of the border of each non-selected boundary
      pType: color
    :param selectedFill:
      helpText: Fill color for components selected in the component table
      pType: color
    :param focusedFill:
      helpText: Fill color for the component currently in the focused image
      pType: color
    :param classColormap:
      helpText: Colormap to use for fill colors by component class
      pType: popuplineeditor
      limits: {cmapVals}
    :param fillAlpha:
      helpText: Transparency of fill color (0 is totally transparent, 1 is totally opaque)
      limits: [0,1]
      step: 0.1
    """
    if len(self.regionData) == 0:
      return
    classes = FR_SINGLETON.tableData.compClasses
    nClasses = len(classes)
    cmap = cm.get_cmap(classColormap, nClasses)
    selectedFill = pg.Color(selectedFill)
    focusedFill = pg.Color(focusedFill)

    penColors = pg.mkPen(color=penColor, width=penWidth)

    focusedFill = np.array(pg.Color(focusedFill).getRgbF())
    selectedFill = np.array(pg.Color(selectedFill).getRgbF())
    # combinedFill = (focusedFill + selectedFill)/2
    fillColors = cmap(self.colorData['compClass'])
    selected = self.colorData['selected'].to_numpy()
    focused = self.colorData['focused'].to_numpy()
    fillColors[selected] = selectedFill
    fillColors[focused] = focusedFill
    # fillColors[focused & selected] = combinedFill
    fillColors[:,-1] = fillAlpha

    self.setPen(penColors)
    self.setBrush([pg.Color(f*255) for f in fillColors])
    self.invalidate()

  def drop(self, ids):
    self.regionData.drop(index=ids, inplace=True)

@FR_SINGLETON.registerGroup(FR_CONSTS.CLS_VERT_IMG)
class FRVertexDefinedImg(pg.ImageItem):
  sigRegionReverted = Signal(object) # new GrayImg
  @classmethod
  def __initEditorParams__(cls):
    cls.fillClr, cls.vertClr = FR_SINGLETON.colorScheme.registerProps(
      cls, [FR_CONSTS.SCHEME_REG_FILL_COLOR, FR_CONSTS.SCHEME_REG_VERT_COLOR])

  def __init__(self):
    super().__init__()
    self.verts = FRComplexVertices()
    FR_SINGLETON.colorScheme.sigParamStateUpdated.connect(lambda: self.setImage(
      lut=self.getLUTFromScheme()))

  def embedMaskInImg(self, toEmbedShape: Tuple[int, int]):
    outImg = np.zeros(toEmbedShape, dtype=bool)
    selfShp = self.image.shape
    outImg[0:selfShp[0], 0:selfShp[1]] = self.image
    return outImg

  @FR_SINGLETON.actionStack.undoable('Modify Focused Region')
  def updateFromVertices(self, newVerts: FRComplexVertices, srcImg: GrayImg=None):
    oldImg = self.image
    oldVerts = self.verts

    self.verts = newVerts.copy()
    if len(newVerts) == 0:
      regionData = np.zeros((1, 1), dtype=bool)
    else:
      if srcImg is None:
        stackedVerts = newVerts.stack()
        regionData = newVerts.toMask(asBool=False)
        # Make vertices full brightness
        regionData[stackedVerts.rows, stackedVerts.cols] = 2
      else:
        regionData = srcImg.copy()

    self.setImage(regionData, levels=[0, 2], lut=self.getLUTFromScheme())
    yield
    self.updateFromVertices(oldVerts, oldImg)

  def updateFromMask(self, newMask: BlackWhiteImg):
    # It is expensive to color the vertices, so only find contours if specified by the user
    oldImg = self.image
    oldVerts = self.verts

    newMask = newMask.astype('uint8')
    if np.array_equal(oldImg>0, newMask):
      # Nothing to do
      return
    verts = FRComplexVertices.fromBwMask(newMask)
    stackedVerts = verts.stack()
    newMask[stackedVerts.rows, stackedVerts.cols] = 2
    self.updateFromVertices(verts, srcImg=newMask)
    return

  def getLUTFromScheme(self):
    lut = [(0, 0, 0, 0)]
    for clr in self.fillClr, self.vertClr:
      lut.append(clr.getRgb())
    return np.array(lut, dtype='uint8')

class FRRegionCopierPlot(pg.PlotCurveItem):
  sigCopyStarted = QtCore.Signal()
  sigCopyStopped = QtCore.Signal()

  def __init__(self, mainImg: imageareas.FREditableImgBase=None, parent=None):
    super().__init__(parent)
    self.active = False
    self.inCopyMode = True
    self.baseData = FRVertices()
    self.regionIds = np.ndarray([])
    self.dataMin = FRVertices()
    self.offset = FRVertices([[0,0]])

    self.setShadowPen(color='k', width=2*self.opts['pen'].width())
    """
    Instead of a customizeable color palette for the copy shape, it is easier to
    have a black outline and white inline color for the shape plot which ensures
    all vertices are visible on any background. However, it is not easy to create
    a multicolored pen in pyqt -- the much simpler solution is to simply create
    a shadow pen, where one has a boundary twice as thick as the other.
    """

    self._connectivity = np.ndarray([], bool)
    mainImg.sigMousePosChanged.connect(self.mainMouseMoved)

  def mainMouseMoved(self, xyPos: FRVertices, _pxColor: np.ndarray):
    if not self.active: return
    newData = self.baseData + xyPos
    self.setData(newData[:,0], newData[:,1], connect=self._connectivity)
    self.offset = xyPos - self.dataMin

  def resetBaseData(self, baseData: List[FRComplexVertices], regionIds: OneDArr):
    allData = FRComplexVertices()
    allConnctivity = []
    for verts in baseData: # each list element represents one component
      plotData, connectivity = stackedVertsPlusConnections(verts)
      allData.append(plotData)
      allConnctivity.append(connectivity)
      if len(connectivity) > 0:
        connectivity[-1] = False
    plotData = allData.stack()
    connectivity = np.concatenate(allConnctivity)

    try:
      allMin = plotData.min(0)
      closestPtIdx  = np.argmin(np.sum(np.abs(plotData - allMin), 1))
      # Guarantees that the mouse will be on the boundary closest to the top left
      self.dataMin = plotData[closestPtIdx]
      # connectivity[addtnlFalseConnectivityIdxs] = False
    except ValueError:
      # When no elements are in the array
      self.dataMin = FRVertices([[0,0]])
    baseData: FRVertices = plotData - self.dataMin
    self.baseData = baseData
    self._connectivity = connectivity
    self.setData(plotData[:,0], plotData[:,1], connect=connectivity)

    self.regionIds = regionIds

  def erase(self):
    self.resetBaseData([FRComplexVertices()], np.array([]))
    self.active = False