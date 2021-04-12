from __future__ import annotations

from typing import Tuple, Sequence, Optional, List

import numpy as np
import pandas as pd
import pyqtgraph as pg
from matplotlib import cm
from matplotlib.pyplot import colormaps
from pandas import DataFrame as df
from pyqtgraph import arrayToQPath
from pyqtgraph.Qt import QtGui, QtCore
from utilitys import PrjParam, RunOpts, EditorPropsMixin, fns

from s3a import PRJ_SINGLETON
from s3a.constants import REQD_TBL_FIELDS as RTF, PRJ_CONSTS, PRJ_ENUMS
from s3a.generalutils import stackedVertsPlusConnections
from s3a.structures import GrayImg, OneDArr, BlackWhiteImg
from s3a.structures import XYVertices, ComplexXYVertices
from . import imageareas
from .clickables import BoundScatterPlot

__all__ = ['MultiRegionPlot', 'VertexDefinedImg', 'RegionCopierPlot']

from .._io import defaultIo

Signal = QtCore.Signal

def makeMultiRegionDf(numRows=1, idList: Sequence[int]=None, selected:Sequence[bool]=None,
                      focused: Sequence[bool]=None,
                      vertices: Sequence[ComplexXYVertices]=None, lblField: PrjParam=None):
  """
  Helper for creating new dataframe holding information determining color data.
  `selected` and `focused` must be boolean arrays indicating whether or not each component
  is selected or focused, respectively.
  If `lblField` is given, it is used as a value to color the components
  """
  outDict = {}
  if selected is None:
    selected = np.zeros(numRows, bool)
  outDict[PRJ_ENUMS.FIELD_SELECTED] = selected
  if focused is None:
    focused = np.zeros(numRows, bool)
  outDict[PRJ_ENUMS.FIELD_FOCUSED] = focused
  if lblField is not None:
    labels_tmp = np.tile(lblField.value, numRows)
    labels = lblField.toNumeric(labels_tmp, rescale=True)
  else:
    labels = np.zeros(numRows)
  outDict[PRJ_ENUMS.FIELD_LABEL] = labels
  if vertices is None:
    vertices = [RTF.VERTICES.value for _ in range(numRows)]
  outDict[RTF.VERTICES] = vertices
  outDf = pd.DataFrame(outDict)
  if idList is not None:
    outDf = outDf.set_index(idList, drop=True)
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

def _makeBoundSymbol(verts: XYVertices):
  verts = verts - verts.min(0, keepdims=True)
  path = arrayToQPath(*verts.T, connect='finite')
  return path

class MultiRegionPlot(EditorPropsMixin, BoundScatterPlot):
  __groupingName__ = PRJ_CONSTS.CLS_MULT_REG_PLT.name

  @classmethod
  def __initEditorParams__(cls):
    super().__initEditorParams__()

  def __init__(self, parent=None):
    super().__init__(size=1, pxMode=False)
    # Wrapping in atomic process means when users make changes to properties, these are maintained when calling the
    # function internally with no parameters
    with PRJ_SINGLETON.colorScheme.setBaseRegisterPath(self.__groupingName__):
      self.updateColors = PRJ_SINGLETON.colorScheme.registerFunc(
        self.updateColors, runOpts=RunOpts.ON_CHANGED, nest=False, ignoreKeys=['hideFocused'])
    self.setParent(parent)
    self.setZValue(50)
    self.regionData = makeMultiRegionDf(0)
    self.cmap = np.array([])
    self.updateColors()

    # 'pointsAt' is an expensive operation if many points are in the scatterplot. Since
    # this will be called anyway when a selection box is made in the main image, disable
    # mouse click listener to avoid doing all that work for nothing.
    # self.centroidPlts.mouseClickEvent = lambda ev: None
    self.mouseClickEvent = lambda ev: None
    # Also disable sigClicked. This way, users who try connecting to this signal won't get
    # code that runs but never triggers
    # self.centroidPlts.sigClicked = None
    self.sigPointsClicked = None

  def resetRegionList(self, newRegionDf: Optional[df]=None,
                      lblField:PrjParam=RTF.INST_ID):
    idList = None
    if (newRegionDf is not None
        and lblField in newRegionDf.columns):
      newRegionDf = newRegionDf.copy()
      newRegionDf[PRJ_ENUMS.FIELD_LABEL] = lblField.toNumeric(newRegionDf[lblField], rescale=True)
    numRows = len(newRegionDf)
    if newRegionDf is not None:
      idList = newRegionDf.index
    self.regionData = makeMultiRegionDf(numRows, idList=idList, lblField=lblField)
    if newRegionDf is not None:
      self.regionData.update(newRegionDf)
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
    for col, idList in zip(self.regionData.columns, [selectedIds, focusedIds]):
      if idList is None: continue
      self.regionData[col] = False
      idList = np.intersect1d(self.regionData.index, idList)
      self.regionData.loc[idList, col] = True
    self.updateColors()

  def updatePlot(self):
    # -----------
    # Update data
    # -----------
    boundLocs = []
    boundSymbs = []
    if self.regionData.empty:
      self.setData(x=[], y=[], data=[])
      return

    for region, _id in zip(self.regionData[RTF.VERTICES],
                           self.regionData.index):
      concatRegion, isfinite = stackedVertsPlusConnections(region)
      boundLoc = np.nanmin(concatRegion, 0, keepdims=True)
      boundSymbol = pg.arrayToQPath(*(concatRegion-boundLoc+0.5).T, connect=isfinite)

      boundLocs.append(boundLoc)
      boundSymbs.append(boundSymbol)

    plotRegions = np.vstack(boundLocs)
    self.setData(*plotRegions.T, symbol=boundSymbs,
                          data=self.regionData.index)
    self.updateColors()

  def toGrayImg(self, imShape: Sequence[int]=None):
    uint16Max = 2**16-2 # Subtract 1 extra so there's room for offset
    labels = (self.regionData[PRJ_ENUMS.FIELD_LABEL]*uint16Max).astype('uint16')
    labelDf = pd.DataFrame()
    labelDf[RTF.VERTICES] = self.regionData[RTF.VERTICES]
    # Override id column to avoid an extra parameter
    labelDf[RTF.INST_ID] = labels
    return defaultIo.exportLblPng(labelDf, imShape=imShape, allowOffset=True)

  @fns.dynamicDocstring(cmapVals=colormaps() + ['None'])
  def updateColors(self, penWidth=0, penColor='w', selectedFill='#00f', focusedFill='#f00', labelColormap='viridis',
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
    :param labelColormap:
      helpText: "Colormap to use for fill colors by component label. If `None` is selected,
        the fill will be transparent."
      pType: popuplineeditor
      limits: {cmapVals}
    :param fillAlpha:
      helpText: Transparency of fill color (0 is totally transparent, 1 is totally opaque)
      limits: [0,1]
      step: 0.1
    :param hideFocused: Many plugins alter the visual behavior of focused regions, so it
      is helpful to have a flag which will cause focused regions to be hidden.
    """
    if len(self.regionData) == 0 or len(self.data) == 0:
      return
    def colorsWithAlpha(_numericLbls: np.ndarray):
      useAlpha = fillAlpha
      if labelColormap == 'None':
        cmap = lambda _classes: np.zeros((len(_classes), 4))
        useAlpha = 0.0
      else:
        cmap = cm.get_cmap(labelColormap)
      colors = cmap(_numericLbls)
      colors[:,-1] = useAlpha
      return colors
    selectedFill = pg.Color(selectedFill)
    focusedFill = pg.Color(focusedFill)

    focusedFill = np.array(pg.Color(focusedFill).getRgbF())
    focusedFill[-1] = fillAlpha
    selectedFill = np.array(pg.Color(selectedFill).getRgbF())
    selectedFill[-1] = fillAlpha
    # combinedFill = (focusedFill + selectedFill)/2
    lbls = self.regionData[PRJ_ENUMS.FIELD_LABEL].to_numpy()
    fillColors = colorsWithAlpha(lbls)
    selected = self.regionData[PRJ_ENUMS.FIELD_SELECTED].to_numpy()
    focused = self.regionData[PRJ_ENUMS.FIELD_FOCUSED].to_numpy()
    fillColors[selected] = selectedFill
    fillColors[focused] = focusedFill
    # fillColors[focused & selected] = combinedFill
    penColors = np.tile(pg.mkPen(color=penColor, width=penWidth), len(lbls))
    self.setPen(penColors)
    self.setBrush([pg.Color(f*255) for f in fillColors])
    self.invalidate()

  def drop(self, ids):
    self.regionData.drop(index=ids, inplace=True)

  def dataBounds(self, ax, frac=1.0, orthoRange=None):
    allVerts = ComplexXYVertices()
    for v in self.regionData[RTF.VERTICES]:
      allVerts.extend(v)
    allVerts = allVerts.stack()
    if len(allVerts) == 0:
      return [None, None]
    bounds = np.r_[allVerts.min(0, keepdims=True)-0.5, allVerts.max(0, keepdims=True)+0.5]
    return list(bounds[:,ax])


class VertexDefinedImg(EditorPropsMixin, pg.ImageItem):
  sigRegionReverted = Signal(object) # new GrayImg

  __groupingName__ = 'Focused Image Graphics'

  @classmethod
  def __initEditorParams__(cls):
    cls.fillClr, cls.vertClr = PRJ_SINGLETON.colorScheme.registerProps(
      [PRJ_CONSTS.SCHEME_REG_FILL_COLOR, PRJ_CONSTS.SCHEME_REG_VERT_COLOR])

  def __init__(self):
    super().__init__()
    self.verts = ComplexXYVertices()
    PRJ_SINGLETON.colorScheme.sigChangesApplied.connect(lambda: self.setImage(
      lut=self.getLUTFromScheme()))

  def embedMaskInImg(self, toEmbedShape: Tuple[int, int]):
    outImg = np.zeros(toEmbedShape, dtype=bool)
    selfShp = self.image.shape
    outImg[0:selfShp[0], 0:selfShp[1]] = self.image
    return outImg

  @PRJ_SINGLETON.actionStack.undoable('Modify Focused Region')
  def updateFromVertices(self, newVerts: ComplexXYVertices, srcImg: GrayImg=None):
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
    verts = ComplexXYVertices.fromBwMask(newMask)
    stackedVerts = verts.stack()
    newMask[stackedVerts.rows, stackedVerts.cols] = 2
    self.updateFromVertices(verts, srcImg=newMask)
    return

  def getLUTFromScheme(self):
    lut = [(0, 0, 0, 0)]
    for clr in self.fillClr, self.vertClr:
      lut.append(clr.getRgb())
    return np.array(lut, dtype='uint8')

class RegionCopierPlot(pg.PlotCurveItem):
  sigCopyStarted = QtCore.Signal()
  sigCopyStopped = QtCore.Signal()

  def __init__(self, mainImg: imageareas.MainImage=None, parent=None):
    super().__init__(parent)
    self.active = False
    self.inCopyMode = True
    self.baseData = XYVertices()
    self.regionIds = np.ndarray([])
    self.dataMin = XYVertices()
    self.offset = XYVertices([[0,0]])

    self.setShadowPen(color='k', width=2*self.opts['pen'].width())
    """
    Instead of a customizeable color palette for the copy shape, it is easier to
    have a black outline and white inline color for the shape plot which ensures
    all vertices are visible on any background. However, it is not easy to create
    a multicolored pen in pyqt -- the much simpler solution is to simply create
    a shadow pen, where one has a boundary twice as thick as the other.
    """

    self._connectivity = np.ndarray([], bool)
    mainImg.sigMouseMoved.connect(self.mainMouseMoved)

  def mainMouseMoved(self, xyPos: np.ndarray):
    if not self.active: return
    newData = self.baseData + xyPos
    self.setData(newData[:,0], newData[:,1], connect=self._connectivity)
    self.offset = xyPos - self.dataMin

  def resetBaseData(self, baseData: List[ComplexXYVertices], regionIds: OneDArr):
    allData = ComplexXYVertices()
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
      self.dataMin = XYVertices([[0,0]])
    baseData: XYVertices = plotData - self.dataMin
    self.baseData = baseData
    self._connectivity = connectivity
    self.setData(plotData[:,0], plotData[:,1], connect=connectivity)

    self.regionIds = regionIds

  def erase(self):
    self.resetBaseData([ComplexXYVertices()], np.array([]))
    self.active = False