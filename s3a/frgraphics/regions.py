from typing import Tuple, Sequence, Optional, Any, Dict, Union

import cv2 as cv
import numpy as np
import pandas as pd
import pyqtgraph as pg
from pandas import DataFrame as df
from pyqtgraph import arrayToQPath
from pyqtgraph.Qt import QtGui, QtCore

from s3a import FR_SINGLETON, appInst
from s3a.generalutils import coerceDfTypes, stackedVertsPlusConnections
from s3a.projectvars import REQD_TBL_FIELDS, FR_CONSTS
from s3a.structures import FRParam, FRVertices, FRComplexVertices, OneDArr, BlackWhiteImg
from s3a.structures.typeoverloads import GrayImg
from .clickables import FRBoundScatterPlot
from .rois import SHAPE_ROI_MAPPING, FRExtendedROI

Signal = QtCore.Signal
Slot = QtCore.Slot

@FR_SINGLETON.registerGroup(FR_CONSTS.CLS_ROI_CLCTN)
class FRShapeCollection(QtCore.QObject):
  # Signal(FRExtendedROI)
  sigShapeFinished = Signal(object)

  @classmethod
  def __initEditorParams__(cls):
    cls.roiClr, cls.roiLineWidth = FR_SINGLETON.scheme.registerProps(cls,
                   [FR_CONSTS.SCHEME_ROI_LINE_CLR, FR_CONSTS.SCHEME_ROI_LINE_WIDTH])

  def __init__(self, allowableShapes: Tuple[FRParam,...]=None, parent: pg.GraphicsView=None):
    super().__init__(parent)
    if allowableShapes is None:
      allowableShapes = set()
    self.shapeVerts = FRVertices()
    # Make a new graphics item for each roi type
    self.roiForShape: Dict[FRParam, Union[pg.ROI, FRExtendedROI]] = {}
    self.forceBlockRois = True

    self._curShape = allowableShapes[0]
    self._allowableShapes = allowableShapes
    self._parent = parent

    for shape in allowableShapes:
      newRoi = SHAPE_ROI_MAPPING[shape]()
      newRoi.setZValue(1000)
      self.roiForShape[shape] = newRoi
      newRoi.hide()
    self.addRoisToView(parent)

    FR_SINGLETON.scheme.sigParamStateUpdated.connect(lambda: self.clearAllRois())

  def addRoisToView(self, view: pg.GraphicsView):
    self._parent = view
    if view is not None:
      for roi in self.roiForShape.values():
        roi.hide()
        view.addItem(roi)

  def clearAllRois(self):
    for roi in self.roiForShape.values():
      while roi.handles:
        # TODO: Submit bug request in pyqtgraph. removeHandle of ROI takes handle or
        #  integer index, removeHandle of PolyLine requires handle object. So,
        #  even though PolyLine should be able  to handle remove by index, it can't
        roi.removeHandle(roi.handles[0]['item'])
      roi.hide()
      roi.pen.setColor(self.roiClr)
      roi.pen.setWidth(self.roiLineWidth)
      self.forceBlockRois = True


  def buildRoi(self, ev: QtGui.QMouseEvent, imgItem: pg.ImageItem=None):
    """
    Construct the current shape ROI depending on mouse movement and current shape parameters
    :param imgItem: Image the ROI is drawn upon, used for mapping event coordinates
      from a scene to pixel coordinates. If *None*, event coordinates are assumed
      to already be relative to pixel coordinates.
    :param ev: Mouse event
    """
    # Unblock on mouse press
    # None imgitem is only the case during programmatic calls so allow this case
    if ((imgItem is None or imgItem.image is not None)
        and ev.type() == ev.MouseButtonPress
        and ev.button() == QtCore.Qt.LeftButton):
      self.forceBlockRois = False
    if self.forceBlockRois: return
    if imgItem is not None:
      posRelToImg = imgItem.mapFromScene(ev.pos())
    else:
      posRelToImg = ev.pos()
    # Form of rate-limiting -- only simulate click if the next pixel is at least one away
    # from the previous pixel location
    xyCoord = FRVertices([[posRelToImg.x(), posRelToImg.y()]], dtype=float)
    curRoi = self.roiForShape[self.curShapeParam]
    constructingRoi, self.shapeVerts = curRoi.updateShape(ev, xyCoord)
    if self.shapeVerts is not None:
      self.sigShapeFinished.emit(self.shapeVerts)

    if not constructingRoi:
      # Vertices from the completed shape are already stored, so clean up the shapes.
      curRoi.hide()
    else:
      # Still constructing ROI. Show it
      curRoi.show()

  @property
  def curShapeParam(self): return self._curShape
  @curShapeParam.setter
  def curShapeParam(self, newShape: FRParam):
    """
    When the shape is changed, be sure to reset the underlying ROIs
    :param newShape: New shape
    :return: None
    """
    # Reset the underlying ROIs for a different shape than we currently are using
    if newShape != self._curShape:
      self.clearAllRois()
    self._curShape = newShape

  @property
  def curShape(self):
      return self.roiForShape[self._curShape]

def makeMultiRegionDf(numRows=1, whichCols=None, idList=None) -> df:
  df_list = []
  if whichCols is None:
    whichCols = (REQD_TBL_FIELDS.INST_ID, REQD_TBL_FIELDS.VERTICES)
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
  @classmethod
  def __initEditorParams__(cls):
    (cls.focusedBoundClr, cls.selectedBoundClr, cls.boundClr, cls.boundWidth) = \
      FR_SINGLETON.scheme.registerProps(
        cls, [FR_CONSTS.SCHEME_FOC_BRUSH_CLR, FR_CONSTS.SCHEME_SEL_BOUND_CLR,
              FR_CONSTS.SCHEME_BOUND_CLR, FR_CONSTS.SCHEME_BOUND_WIDTH])


  def __init__(self, parent=None):
    super().__init__(brush=None, size=1, pxMode=False)
    self.setParent(parent)
    self.setZValue(50)
    self.regionData = makeMultiRegionDf(0)
    self.selectedIds = np.array([], dtype=int)
    self.focusedIds = np.array([], dtype=int)


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
    self.regionData = makeMultiRegionDf(0)
    self[newIds,newRegionDf.columns] = newRegionDf

  def selectById(self, selectedIds: OneDArr):
    """
    Marks 'selectedIds' as currently selected by changing their scheme to user-specified
    selection values.
    """
    if len(self.regionData) == 0:
      return
    self.selectedIds = selectedIds
    defaultPen = pg.mkPen(width=self.boundWidth, color=self.boundClr)
    newPens = np.array([defaultPen]*len(self.regionData))
    selectionPen = pg.mkPen(width=self.boundWidth*2, color=self.selectedBoundClr)
    newPens[np.isin(self.regionData.index, selectedIds)] = selectionPen
    self.setPen(newPens)
    self.invalidate()


  def focusById(self, focusedIds: OneDArr):
    """
    Colors 'focusedIds' to indicate they are present in a focused view.
    """
    if len(self.regionData) == 0:
      return
    self.focusedIds = focusedIds
    brushes = np.array([None]*len(self.regionData))

    brushes[np.isin(self.regionData.index, focusedIds)] = pg.mkBrush(self.focusedBoundClr)
    self.setBrush(brushes)
    self.invalidate()


  def updatePlot(self):
    # -----------
    # Update data
    # -----------
    boundLocs = []
    boundSymbs = []
    if self.regionData.empty:
      self.setData(x=[], y=[], data=[])
      return

    for region, _id in zip(self.regionData.loc[:, REQD_TBL_FIELDS.VERTICES],
                           self.regionData.index):
      concatRegion, isfinite = stackedVertsPlusConnections(region)
      boundLoc = np.nanmin(concatRegion, 0, keepdims=True)
      boundSymbol = pg.arrayToQPath(*(concatRegion-boundLoc).T, connect=isfinite)

      boundLocs.append(boundLoc)
      boundSymbs.append(boundSymbol)

    plotRegions = np.vstack(boundLocs)
    width = self.boundWidth
    boundPen = pg.mkPen(color=self.boundClr, width=width)
    self.setData(*plotRegions.T, pen=boundPen, symbol=boundSymbs,
                          data=self.regionData.index)
    self.selectById(self.selectedIds)
    self.focusById(self.focusedIds)

  def __getitem__(self, keys: Tuple[Any,...]):
    """
    Allows retrieval of vertex/valid list for a given set of IDs
    """
    return self.regionData.loc[keys[0], keys[1:]]

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

  def drop(self, ids):
    self.regionData.drop(index=ids, inplace=True)

@FR_SINGLETON.registerGroup(FR_CONSTS.CLS_VERT_IMG)
class FRVertexDefinedImg(pg.ImageItem):
  sigRegionReverted = Signal(object) # new GrayImg
  @classmethod
  def __initEditorParams__(cls):
    cls.fillClr, cls.vertClr = FR_SINGLETON.scheme.registerProps(
      cls, [FR_CONSTS.SCHEME_REG_FILL_COLOR, FR_CONSTS.SCHEME_REG_VERT_COLOR])

  def __init__(self):
    super().__init__()
    self.verts = FRComplexVertices()

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