from typing import Tuple, Sequence, Optional, Any, Dict, Union

import cv2 as cv
import numpy as np
import pandas as pd
import pyqtgraph as pg
from pandas import DataFrame as df
from pyqtgraph.Qt import QtGui, QtCore

from cdef import FR_SINGLETON
from cdef.generalutils import coerceDfTypes, nanConcatList
from cdef.processingutils import getVertsFromBwComps
from cdef.projectvars import REQD_TBL_FIELDS, FR_CONSTS
from cdef.structures import FRParam, FRVertices, FRComplexVertices, OneDArr, BlackWhiteImg
from cdef.structures.typeoverloads import GrayImg
from .clickables import FRCentroidScatterItem
from .rois import SHAPE_ROI_MAPPING, FRExtendedROI

Signal = QtCore.pyqtSignal
Slot = QtCore.pyqtSlot


class FRShapeCollection(QtCore.QObject):
  # Signal(FRExtendedROI)
  sigShapeFinished = Signal(object)
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
      self.forceBlockRois = True


  def buildRoi(self, imgItem: pg.ImageItem, ev: QtGui.QMouseEvent):
    """
        Construct the current shape ROI depending on mouse movement and current shape parameters
        :param imgItem: Image the ROI is drawn upon. Either focused imgItem or main imgItem
        :param ev: Mouse event
        """
    # Unblock on mouse press
    if (imgItem.image is not None
        and ev.type() == ev.MouseButtonPress
        and ev.button() == QtCore.Qt.LeftButton):
      self.forceBlockRois = False
    if self.forceBlockRois: return
    posRelToImg = imgItem.mapFromScene(ev.pos())
    # Form of rate-limiting -- only simulate click if the next pixel is at least one away
    # from the previous pixel location
    xyCoord = FRVertices([[posRelToImg.x(), posRelToImg.y()]], dtype=float)
    curRoi = self.roiForShape[self.curShape]
    constructingRoi, self.shapeVerts = curRoi.updateShape(ev, xyCoord)
    if self.shapeVerts is not None:
      self.sigShapeFinished.emit(curRoi)

    if not constructingRoi:
      # Vertices from the completed shape are already stored, so clean up the shapes.
      curRoi.hide()
    else:
      # Still constructing ROI. Show it
      curRoi.show()

  @property
  def curShape(self): return self._curShape
  @curShape.setter
  def curShape(self, newShape: FRParam):
    """
    When the shape is changed, be sure to reset the underlying ROIs
    :param newShape: New shape
    :return: None
    """
    # Reset the underlying ROIs for a different shape than we currently are using
    if newShape != self._curShape:
      self.clearAllRois()
    self._curShape = newShape

def makeMultiRegionDf(numRows=1, whichCols=None, idList=None) -> df:
  df_list = []
  if whichCols is None:
    whichCols = (REQD_TBL_FIELDS.INST_ID, REQD_TBL_FIELDS.VERTICES, REQD_TBL_FIELDS.VALIDATED)
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

@FR_SINGLETON.registerGroup(FR_CONSTS.CLS_MULT_REG_PLT)
class FRMultiRegionPlot(QtCore.QObject):
  @classmethod
  def __initEditorParams__(cls):
    (cls.nonvalidIdClr, cls.validIdClr, cls.boundClr,cls.boundWidth,
    cls.idMarkerSz, cls.selectedIdBorder) = FR_SINGLETON.scheme.registerProps(cls,
      [FR_CONSTS.SCHEME_NONVALID_ID_COLOR, FR_CONSTS.SCHEME_VALID_ID_COLOR,
       FR_CONSTS.SCHEME_BOUNDARY_COLOR, FR_CONSTS.SCHEME_BOUNDARY_WIDTH,
       FR_CONSTS.SCHEME_ID_MARKER_SZ, FR_CONSTS.SCHEME_SELECTED_ID_BORDER]
    )


  # Helper class for IDE assistance during dataframe access
  def __init__(self, parent=None):
    super().__init__(parent)
    self.boundPlt = pg.PlotDataItem(connect='finite')
    self.centroidPlts = FRCentroidScatterItem(pen=None)
    self.data = makeMultiRegionDf(0)

    # 'pointsAt' is an expensive operation if many points are in the scatterplot. Since
    # this will be called anyway when a selection box is made in the main image, disable
    # mouse click listener to avoid doing all that work for nothing.
    self.centroidPlts.mouseClickEvent = lambda ev: None
    # Also disable sigClicked. This way, users who try connecting to this signal won't get
    # code that runs but never triggers
    self.centroidPlts.sigClicked = None

  def resetRegionList(self, newIds: Optional[Sequence]=None, newRegionDf: Optional[df]=None):
    if newIds is None:
      newIds = []
    if newRegionDf is None:
      newRegionDf = makeMultiRegionDf(0)
    self.data = makeMultiRegionDf(0)
    self[newIds,newRegionDf.columns] = newRegionDf

  def selectById(self, selectedIds: OneDArr):
    """
    Marks 'selectedIds' as currently selected by changing their scheme to user-specified
    selection values.
    """
    selectedIdPens = np.empty(len(self.data), dtype=object)
    selectedIdPens.fill(None)
    selectedIdxs = np.isin(self.data.index, selectedIds)
    selectedIdPens[selectedIdxs] = pg.mkPen(self.selectedIdBorder, width=3)

    self.centroidPlts.setPen(selectedIdPens)

  def focusById(self, focusedIds: OneDArr):
    """
    Colors 'focusedIds' to indicate they are present in a focused view.
    """
    focusedIdSymbs = np.empty(len(self.data), dtype='<U4')
    focusedIdSymbs.fill('o')
    focusedIdSizes = np.empty(len(self.data))
    focusedIdSizes.fill(self.idMarkerSz)
    focusedIdxs = np.isin(self.data.index, focusedIds)

    # TODO: Make GUI properties for these?
    focusedIdSymbs[focusedIdxs] = 'star'
    focusedIdSizes[focusedIdxs] *= 3
    self.centroidPlts.setSymbol(focusedIdSymbs)
    self.centroidPlts.setSize(focusedIdSizes)


  def updatePlot(self):
    # -----------
    # Update data
    # -----------
    plotRegions = [np.ones((0,2))]
    idLocs = [np.ones((0,2))]

    for region in self.data.loc[:, REQD_TBL_FIELDS.VERTICES]:
      concatRegion = nanConcatList(region)
      idLoc = np.nanmean(concatRegion, 0)
      idLocs.append(idLoc)
      # Before stacking regions, add first point of region to end of region vertices.
      # This will make the whole region connected in the output plot
      # Insert nan to make separate components unconnected
      plotRegions.append(concatRegion)
    idLocs = np.vstack(idLocs)
    # TODO: If the 'development' branch of pyqtgraph is set up, the clickable portion of each
    #   plot can be the ID of the component. Otherwise it must be a non-descript item.
    #scatSymbols = [_makeTxtSymbol(str(curId), idSz) for curId in self.data.index]
    scatSymbols = [None]*len(self.data)

    brushes = np.empty(len(self.data), dtype=object)
    brushes.fill(pg.mkBrush(self.nonvalidIdClr))
    brushes[self.data.loc[:, REQD_TBL_FIELDS.VALIDATED]] = pg.mkBrush(self.validIdClr)

    self.centroidPlts.setData(x=idLocs[:, 0], y=idLocs[:, 1], size=self.idMarkerSz, brush=brushes,
                              data=self.data.index, symbol=scatSymbols)
    plotRegions = np.vstack(plotRegions)
    boundPen = pg.mkPen(color=self.boundClr, width=self.boundWidth)
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
    # Since we may only be resetting one parameter (either valid or regions),
    # Make sure to keep the old parameter value for the unset index
    keysDf.update(self.data)
    keysDf.loc[regionIds, setVals] = vals
    self.data.update(keysDf)

    # Now we can add entries that weren't in our original dataframe
    # If not all set values were provided in the new dataframe, fix this by embedding
    # it into the default dataframe
    newDataDf = makeMultiRegionDf(np.sum(newEntryIdxs), idList=regionIds[newEntryIdxs])
    newDataDf.loc[:, keysDf.columns] = keysDf.loc[newEntryIdxs, :]
    self.data = pd.concat((self.data, newDataDf))
    # Retain type information
    coerceDfTypes(self.data, makeMultiRegionDf(0).columns)
    self.updatePlot()

  def drop(self, ids):
    self.data.drop(index=ids, inplace=True)

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
    idxs = tuple(slice(0, self.image.shape[ii]) for ii in range(2))
    outImg[idxs] = self.image
    return outImg

  @FR_SINGLETON.actionStack.undoable('Modify Focused Region')
  def updateFromVertices(self, newVerts: FRComplexVertices, srcImg: GrayImg=None):
    oldImg = self.image
    oldVerts = self.verts

    self.verts = newVerts.copy()
    if len(newVerts.x_flat) == 0:
      regionData = np.zeros((1, 1), dtype=bool)
    else:
      if srcImg is None:
        newImgShape = newVerts.stack().max(0)[::-1] + 1
        regionData = np.zeros(newImgShape, dtype='uint8')
        cv.fillPoly(regionData, newVerts, 1)
        # Make vertices full brightness
        regionData[newVerts.y_flat, newVerts.x_flat] = 2
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
    verts = getVertsFromBwComps(newMask)
    newMask[verts.y_flat, verts.x_flat] = 2
    self.updateFromVertices(verts)
    return

  def getLUTFromScheme(self):
    lut = [(0, 0, 0, 0)]
    for clr in self.fillClr, self.vertClr:
      lut.append(clr.getRgb())
    return np.array(lut, dtype='uint8')