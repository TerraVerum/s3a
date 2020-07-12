from typing import Tuple, Dict, Union

import pyqtgraph as pg
from PyQt5 import QtCore, QtGui

from s3a import FR_SINGLETON
from s3a.projectvars import FR_CONSTS
from s3a.structures import FRParam, FRVertices
from s3a.views.rois import FRExtendedROI, SHAPE_ROI_MAPPING

__all__ = ['FRShapeCollection']

@FR_SINGLETON.registerGroup(FR_CONSTS.CLS_ROI_CLCTN)
class FRShapeCollection(QtCore.QObject):
  # Signal(FRExtendedROI)
  sigShapeFinished = QtCore.Signal(object)

  @classmethod
  def __initEditorParams__(cls):
    cls.roiClr, cls.roiLineWidth = FR_SINGLETON.scheme.registerProps(cls,
                   [FR_CONSTS.SCHEME_ROI_LINE_CLR, FR_CONSTS.SCHEME_ROI_LINE_WIDTH])

  def __init__(self, allowableShapes: Tuple[FRParam,...]=(), parent: pg.GraphicsView=None):
    super().__init__(parent)
    if allowableShapes is None:
      allowableShapes = set()
    self.shapeVerts = FRVertices()
    # Make a new graphics item for each roi type
    self.roiForShape: Dict[FRParam, Union[pg.ROI, FRExtendedROI]] = {}
    self.forceBlockRois = True

    self._curShape = allowableShapes[0] if len(allowableShapes) > 0 else None
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