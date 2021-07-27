from inspect import isdatadescriptor
import typing as t

import numpy as np
from pyqtgraph import SignalProxy
from utilitys import ParamEditorPlugin
from utilitys.fns import serAsFrame
from utilitys.widgets import ImageViewer

from ..logger import getAppLogger
from ..shared import SharedAppSettings
from utilitys import ParamContainer
from ..constants import PRJ_CONSTS as CNST
from ..views.imageareas import MainImage
import pandas as pd
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
from datetime import datetime, timedelta
from pyqtgraph.Qt import isQObjectAlive

METRIC_TICK_INTERVAL_S = 0.25

class MetricsEventFilter(QtCore.QObject):
  """
  Functions as the event filter to grab mouse information
  """
  sigMouseMoved = QtCore.Signal(object)
  """dict with [x, y] pos, current action, device pixel size"""

  def __init__(self, parent=None, metrics=None):
    super().__init__(parent)
    self.metrics = metrics

  def eventFilter(self, watched:QtCore.QObject, event:QtCore.QEvent):
    event: QtGui.QMouseEvent
    mImg = self.metrics.win.mainImg
    if not (QtWidgets.QApplication.mouseButtons() == QtCore.Qt.MouseButton.LeftButton
            and event.type() in [event.Type.MetaCall]
            and mImg.drawAction in [CNST.DRAW_ACT_CREATE, CNST.DRAW_ACT_ADD, CNST.DRAW_ACT_REM]
    ):
      return False
    globalPos = QtGui.QCursor.pos()
    scenePos = mImg.mapFromGlobal(globalPos)
    itemPos = mImg.imgItem.mapFromScene(scenePos)
    # Aspect ratio is locked, so pixel width and height are the same
    toEmit = dict(action=mImg.drawAction, mouse_pos=(itemPos.x(), itemPos.y()),
                  pixel_size=mImg.imgItem.pixelWidth())
    self.sigMouseMoved.emit(toEmit)
    return False

class UserMetricsPlugin(ParamEditorPlugin):
  name = 'User Metrics'

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.collectedMetrics = pd.DataFrame(
      columns=['time', 'viewbox_range', 'mouse_pos', 'action', 'pixel_size']
    ).set_index(['time'])
    self.mainImgMouseFilter = MetricsEventFilter(metrics=self)
    self._metricsImage = None
    self._metricsViewer = self._metricsViewerContainer = None
    self.collectorProxies = []
    self.registerFunc(self.showMetricsWidget)
    self.registerFunc(self.resetMetrics)

  def __initEditorParams__(self, shared: SharedAppSettings):
    super().__initEditorParams__(shared=shared)
    self.props = ParamContainer()
    param = shared.generalProps.registerProp(
      CNST.PROP_COLLECT_USR_METRICS,
      container=self.props, asProperty=False)
    def onPropChange(param, value):
      self.menu.menuAction().setVisible(value)
      if value:
        self.activateMetricCollection()
      else:
        self.deactivateMetricCollection()
    param.sigValueChanged.connect(onPropChange)

  def attachWinRef(self, win):
    super().attachWinRef(win)
    
    if self.props[CNST.PROP_COLLECT_USR_METRICS]:
      self.activateMetricCollection()
    else:
      self.menu.menuAction().setVisible(False)

    win.mainImg.imgItem.sigImageChanged.connect(self.resetMetrics)
    

  def activateMetricCollection(self):
    mImg: MainImage = self.win.mainImg
    def collectViewboxMetrics(args):
      vb, vbRange, axes = args
      self.updateUserMetrics(viewbox_range=vbRange)
    def collectMouseMetrics(metricsDict):
      # Receives a tuple for some reason?
      self.updateUserMetrics(**metricsDict[0])
    self.collectorProxies = [
      SignalProxy(mImg.getViewBox().sigRangeChanged,
                  slot=collectViewboxMetrics,
                  rateLimit=int(1/METRIC_TICK_INTERVAL_S)),
      SignalProxy(self.mainImgMouseFilter.sigMouseMoved,
                  slot=collectMouseMetrics,
                  rateLimit=int(1/METRIC_TICK_INTERVAL_S))
    ]

    mImg.scene().installEventFilter(self.mainImgMouseFilter)
  
  def deactivateMetricCollection(self):
    mImg: MainImage = self.win.mainImg
    mImg.scene().removeEventFilter(self.mainImgMouseFilter)
    for proxy in self.collectorProxies:
      proxy.disconnect
    self.collectorProxies = []

  def updateUserMetrics(self, **kwargs):
    """
    Updates list of obtained metrics with those found in kwargs
    """
    getAppLogger(__name__).debug(f'Recorded stats: {kwargs}')
    time = kwargs.pop('time', datetime.now())
    toInsert = pd.Series(kwargs, name=time)
    self.collectedMetrics = self.collectedMetrics.append(toInsert)
    
    lowerBound = 1 if 'viewbox_range' in kwargs else 2
    self.incrementUserMetricsImage(n=lowerBound)

  def incrementUserMetricsImage(self, n=1):
    """
    Updates the current metrics image instead of recreating it by applying the last n rows
    of collected metrisc
    """
    if not len(self.collectedMetrics):
      useMetrics = None
    else:
      useMetrics = self.collectedMetrics.iloc[-n:]
    self._metricsImage = self.imageFromMetrics(useMetrics)
    if self._metricsViewer is not None and isQObjectAlive(self._metricsViewer):
      self._metricsViewer.setImage(self._metricsImage)


  def imageFromMetrics(self, metricsToUse=None):
    actions = [CNST.DRAW_ACT_REM, CNST.DRAW_ACT_ADD, CNST.DRAW_ACT_CREATE]

    if metricsToUse is None:
      metricsToUse = self.collectedMetrics
      image = None
    else:
      image = self._metricsImage
    if image is None:
      image = np.zeros((*self.win.mainImg.image.shape[:2], len(actions)))
        
    vbRanges = metricsToUse.loc[metricsToUse['viewbox_range'].notnull(),
                                         'viewbox_range']
    if len(vbRanges):
      self._populateViewboxVotes(vbRanges, image)
    for ii, act in enumerate(actions):
      # Mouse actions should be interpreted differently depending on the action
      # Mouse also uses diff, so make sure at least two votes exist
      metrics = metricsToUse.loc[metricsToUse['action'] == act, ['mouse_pos', 'pixel_size']]
      metrics = metrics.dropna()
      if len(metrics) < 2:
        continue
      self._populateMouseVotes(metrics, image[...,ii])
    return image

  @staticmethod
  def _populateViewboxVotes(votes: pd.Series, image: np.ndarray):
    """
    Records viewbox range votes from collected metrics into the image
    :param votes: Viewbox ranges from metrics
    :param image: Image to record votes into
    """
    for vb in votes:
      # Viewbox has float values, convert to int for indexing
      vb = np.asarray(vb)
      # Negative values index by wrapping around the image, but these should be clipped to 0 instead
      # No need to clip high values, since these won't be considered during the indexing anyway
      # if they're too large
      vb = np.clip(vb, 0, np.inf).astype(int)
      (x1, x2), (y1, y2) = vb
      image[y1:y2, x1:x2, ...] += 1

  @staticmethod
  def _populateMouseVotes(votes: pd.DataFrame, image: np.ndarray, maxVoteWeight=5, distToPxSizeRatio=0.5):
    """
    Records votes from mouse movement into image. Slower movement is considered a
    higher vote
    :param votes: Mouse positions from metrics and pixel sizes
    :param image: Image to record votes into
    """
    xyCoords = np.row_stack(votes['mouse_pos'])
    diffs = np.abs(np.diff(xyCoords, axis=0))
    dists = diffs[:,0] + diffs[:,1]
    # Since slower movement is a smaller distance number, and we want the opposite to be true,
    # find the inverse but protect against a divide by zero
    dists[dists == 0] = 0.01
    distWgt = 1-distToPxSizeRatio
    wgts = np.clip(dists.max()/dists, 0, maxVoteWeight) * distWgt
    # Since diff is used, the first coordinate must be discarded
    xyCoords = xyCoords[1:].astype(int)
    pxSizes = np.round(votes['pixel_size'])[1:].astype(int)
    # Dilate each coordinate by the pixel size to account for potential variances
    # The smaller the device pixel size, the stronger the confidence/weight should be too
    for coord, pxSize, wgt in zip(xyCoords, pxSizes, wgts):
      (x, y) = coord
      pxSize = max(1, pxSize)
      pxWeight = maxVoteWeight*distToPxSizeRatio/pxSize
      # Expand the image by the pixel size
      image[y-pxSize:y+pxSize+1, x-pxSize:x+pxSize+1, ...] += wgt + pxWeight

  def showMetricsWidget(self):
    if (self._metricsViewer is not None
        and isQObjectAlive(self._metricsViewer)
        and self._metricsViewerContainer is not None
        and isQObjectAlive(self._metricsViewerContainer)
    ):
      if not self._metricsViewerContainer.isVisible():
        self._metricsViewerContainer.show()
      return
    viewer = ImageViewer(self._metricsImage)
    container = viewer.widgetContainer()
    container.show()
    self._metricsViewer = viewer
    # Keep reference to container to prevent garbage collection
    self._metricsViewerContainer = container
    container.show()

  def resetMetrics(self):
    self.collectedMetrics = self.collectedMetrics.iloc[0:0].copy()
    self.incrementUserMetricsImage()