from typing import Callable, Dict

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from skimage.draw import draw

from s3a.generalutils import orderContourPts, symbolFromVerts
from utilitys import PrjParam

from s3a.constants import PRJ_CONSTS
from s3a.structures import XYVertices, ComplexXYVertices

__all__ = ['RectROI', 'PlotDataROI', 'PolygonROI', 'PointROI', 'SHAPE_ROI_MAPPING']
from s3a.views.clickables import BoundScatterPlot

qe = QtCore.QEvent


class PlotDataROI(BoundScatterPlot):
  connected = True

  def __init__(self, endEvType=qe.MouseButtonRelease, addIntermedPts=True):
    super().__init__(pxMode=False, size=1)
    self.setBrush(pg.mkBrush(0, 0, 0, 100))
    self.setPen(pg.mkPen('w', width=0))
    self.vertices = XYVertices()

    self.endEvType = endEvType
    self.viableEvTypes = {
      qe.MouseButtonPress, qe.MouseMove, qe.MouseButtonRelease, qe.MouseButtonDblClick
    } - {endEvType}
    self.constructingRoi = False
    self.addIntermedPts = addIntermedPts
    self.firstPt = XYVertices()

  def addRoiPoints(self, pts: XYVertices):
    # noinspection PyTypeChecker
    pts = pts.astype(int)
    if self.vertices.size > 0 and np.all(pts == self.vertices[-1]):
      return
    if self.addIntermedPts or len(self.vertices) < 2:
      vertsToUse = np.vstack([self.vertices, pts])
    else:
      vertsToUse = np.vstack([self.firstPt, pts])
    refactored = self._refactorPoints(vertsToUse)
    verts = self.vertices = XYVertices(refactored)
    connectData = np.vstack([verts, verts[[0]]]).view(XYVertices)
    symbol, pos = symbolFromVerts(ComplexXYVertices([connectData]))
    self.setData(*pos.T, symbol=[symbol])


  def setRoiPoints(self, pts: XYVertices=None):
    if pts is None:
      pts = XYVertices()
    self.setData()
    self.vertices = XYVertices()
    self.firstPt = pts
    if pts.size > 0:
      self.addRoiPoints(pts)

  def _refactorPoints(self, vertices: np.ndarray):
    return vertices

  def updateShape(self, ev: QtGui.QMouseEvent, xyEvCoords: XYVertices):
    """
    See function signature for :func:`ExtendedROI.updateShape`
    """
    success = True
    verts = None
    constructingRoi = self.constructingRoi
    # If not left click, do nothing
    if (int(ev.buttons()) & QtCore.Qt.LeftButton) == 0 \
        and ev.button() != QtCore.Qt.LeftButton:
      return self.constructingRoi, verts
    evType = ev.type()
    if evType == qe.MouseButtonPress and not constructingRoi:
      # Need to start a new shape
      self.setRoiPoints(xyEvCoords)
      constructingRoi = True
    if evType in self.viableEvTypes:
      self.addRoiPoints(xyEvCoords)
      constructingRoi = True
    elif evType == self.endEvType:
      # Done drawing the ROI, complete shape, get vertices
      verts = self.vertices
      constructingRoi = False
    else:
      success = False
      # Unable to process this event

    if success:
      ev.accept()
    self.constructingRoi = constructingRoi
    return self.constructingRoi, verts

class RectROI(PlotDataROI):

  def __init__(self, endEvType=qe.MouseButtonRelease):
    super().__init__(endEvType)
    self.addIntermedPts = False

  def _refactorPoints(self, vertices: np.ndarray):
    square = XYVertices(np.vstack([vertices.min(0), vertices.max(0)]))
    return np.vstack([
      square[0],
      [square[0,0], square[1,1]],
      [square[1,0], square[1,1]],
      [square[1,0], square[0,1]]
    ])

class PolygonROI(PlotDataROI):

  def __init__(self):
    super().__init__(qe.MouseButtonDblClick, True)
    self.viableEvTypes.remove(qe.MouseMove)
    self.viableEvTypes.remove(qe.MouseButtonRelease)

class PointROI(PlotDataROI):
  roiRadius = 1

  @classmethod
  def updateRadius(cls, radius=1):
    cls.roiRadius = radius


  def __init__(self):
    super().__init__()
    self.setSize(3)
    self.constructTypes = {qe.MouseButtonPress, qe.MouseMove}

  def updateShape(self, ev: QtGui.QMouseEvent, xyEvCoords: XYVertices):
    success = False
    verts = None
    constructingRoi = False
    if ev.type() in self.constructTypes:
      success = True
      self.setData(*xyEvCoords.T, size=self.roiRadius*2)
      verts = XYVertices(np.column_stack(draw.disk(xyEvCoords[0], self.roiRadius)))
      self.vertices = verts
      constructingRoi = True

    if success:
      ev.accept()
    return constructingRoi, verts

class EllipseROI(PlotDataROI):

  def __init__(self):
    super().__init__(endEvType=qe.MouseButtonRelease, addIntermedPts=False)

  def _refactorPoints(self, vertices: np.ndarray):
    pts = vertices[:, ::-1].astype(float)
    ma = pts.max(0)
    mi = pts.min(0)
    center = (ma - mi)/2 + mi
    normedR = ma[0]-mi[0]
    normedC = ma[1]-mi[1]
    # Apply scaling so mouse point is on perimeter
    # angle = np.arctan(normedR/normedC)
    # normedR = abs(np.cos(angle))*normedR
    # normedC = abs(np.sin(angle))*normedC
    sqr2 = np.sqrt(2)
    perim = draw.ellipse_perimeter(*center.astype(int), int(normedR/sqr2), int(normedC/sqr2))
    # Reorder to ensure no criss-crossing when these vertices are plotted
    perim = orderContourPts(np.column_stack(perim[::-1]))
    return perim.view(XYVertices)

SHAPE_ROI_MAPPING: Dict[PrjParam, Callable[[], PlotDataROI]] = {
  PRJ_CONSTS.DRAW_SHAPE_RECT: RectROI,
  PRJ_CONSTS.DRAW_SHAPE_FREE: PlotDataROI,
  PRJ_CONSTS.DRAW_SHAPE_POLY: PolygonROI,
  PRJ_CONSTS.DRAW_SHAPE_ELLIPSE: EllipseROI,
  PRJ_CONSTS.DRAW_SHAPE_POINT: PointROI,
}
