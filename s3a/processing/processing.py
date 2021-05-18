from __future__ import annotations

import typing as t

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from utilitys.processing import *

__all__ = ['ProcessIO', 'ProcessStage', 'NestedProcess', 'ImageProcess',
           'AtomicProcess', 'MultiPredictionProcess']

_infoType = t.List[t.Union[t.List, t.Dict[str, t.Any]]]
StrList = t.List[str]
StrCol = t.Collection[str]

class ImageProcess(NestedProcess):
  inMap = ['image']
  outMap = ['image']

  @classmethod
  def _cmpPrevCurInfos(cls, prevInfos: t.List[dict], infos: t.List[dict]):
    validInfos = super()._cmpPrevCurInfos(prevInfos, infos)
    # Iterate backwards to facilitate entry deletion
    for ii in range(len(validInfos)-1,-1,-1):
      info = validInfos[ii]
      duplicateKeys = {'name'}
      for k, v in info.items():
        if v is cls._DUPLICATE_INFO:
          duplicateKeys.add(k)
      if len(info.keys() - duplicateKeys) == 0:
        del validInfos[ii]
    return validInfos


  def _stageSummaryWidget(self):
    infoToDisplay = self.getStageInfos()

    numStages = len(infoToDisplay)
    nrows = np.sqrt(numStages).astype(int)
    ncols = np.ceil(numStages/nrows)
    outGrid = pg.GraphicsLayoutWidget()
    sizeToAxMapping: t.Dict[tuple, pg.PlotItem] = {}
    for ii, info in enumerate(infoToDisplay):
      pltItem: pg.PlotItem = outGrid.addPlot(title=info.get('name', None))
      vb = pltItem.getViewBox()
      vb.invertY(True)
      vb.setAspectLocked(True)
      npImg = info['image']
      imShp = npImg.shape[:2]
      margin = np.array(imShp)*2
      lim = max(margin + imShp)
      vb.setLimits(maxXRange=lim, maxYRange=lim)
      sameSizePlt = sizeToAxMapping.get(imShp, None)
      if sameSizePlt is not None:
        pltItem.setXLink(sameSizePlt)
        pltItem.setYLink(sameSizePlt)
      sizeToAxMapping[imShp] = pltItem
      imgItem = pg.ImageItem(npImg)
      pltItem.addItem(imgItem)

      if ii % ncols == ncols-1:
        outGrid.nextRow()
    # See https://github.com/pyqtgraph/pyqtgraph/issues/1348. strange zooming occurs
    # if aspect is locked on all figures
    # for ax in sizeToAxMapping.values():
    #   ax.getViewBox().setAspectLocked(True)
    oldClose = outGrid.closeEvent
    def newClose(ev):
      del _winRefs[outGrid]
      oldClose(ev)
    oldResize = outGrid.resizeEvent
    def newResize(ev):
      oldResize(ev)
      for ax in sizeToAxMapping.values():
        ax.getViewBox().autoRange()

    # Windows that go out of scope get garbage collected. Prevent that here
    _winRefs[outGrid] = outGrid
    outGrid.closeEvent = newClose
    outGrid.resizeEvent = newResize

    return outGrid

  def getStageInfos(self, ignoreDuplicates=True):
    infos = super().getStageInfos(ignoreDuplicates)
    # Add entry for initial image since it will be missed when just searching for
    # stage outputs
    infos.insert(0, {'name': 'Initial Image', 'image': self.input['image']})
    return infos

_winRefs = {}

class MultiPredictionProcess(ImageProcess):
  def _stageSummaryWidget(self):
    return QtWidgets.QWidget()
  outMap = ['components']

class CategoricalProcess(NestedProcess):
  def _stageSummaryWidget(self):
    pass

  inMap = ['categories', 'confidences']