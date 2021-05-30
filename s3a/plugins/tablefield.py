from __future__ import annotations

import copy
from collections import deque, namedtuple
from typing import Sequence
from warnings import warn

import numpy as np
import pandas as pd
import pyqtgraph as pg

import s3a
from s3a import PRJ_CONSTS as CNST, XYVertices, REQD_TBL_FIELDS as RTF, ComplexXYVertices
from s3a.processing.algorithms.imageproc import _historyMaskHolder
from s3a.structures import BlackWhiteImg
from s3a.views.regions import MultiRegionPlot, makeMultiRegionDf
from utilitys import PrjParam, DeferredActionStackMixin as DASM
from .base import TableFieldPlugin
from ..constants import PRJ_ENUMS
from ..generalutils import getCroppedImg, showMaskDiff
from ..graphicsutils import RegionHistoryViewer
from ..shared import SharedAppSettings


class _REG_ACCEPTED: pass
buffEntry = namedtuple('buffentry', 'id_ vertices')

class VerticesPlugin(DASM, TableFieldPlugin):
  name = 'Vertices'

  def __initEditorParams__(self, shared: SharedAppSettings):
    super().__initEditorParams__()
    self.procEditor = shared.imgProcClctn.createProcessorEditor(type(self), self.name + ' Processor')

    self.dock.addEditors([self.procEditor])

  def __init__(self):
    super().__init__()
    self.region = MultiRegionPlot()
    self.region.hide()
    self.firstRun = True
    self.playbackWindow = RegionHistoryViewer()
    self.regionBuffer = deque(maxlen=CNST.PROP_UNDO_BUF_SZ.value)

  def attachWinRef(self, win: s3a.S3A):
    win.mainImg.addItem(self.region)

    def resetRegBuff(_, newSize):
      newBuff = deque(maxlen=newSize)
      newBuff.extend(self.regionBuffer)
      self.regionBuffer = newBuff
    mainBufSize = win.sharedAttrs.generalProps.params.child(win.__groupingName__, 'maxLength')
    mainBufSize.sigValueChanged.connect(resetRegBuff)

    funcLst = [self.resetFocusedRegion, self.fillRegionMask, self.clearFocusedRegion, self.clearProcessorHistory]
    paramLst = [CNST.TOOL_RESET_FOC_REGION, CNST.TOOL_FILL_FOC_REGION,
                CNST.TOOL_CLEAR_FOC_REGION, CNST.TOOL_CLEAR_HISTORY]
    for func, param in zip(funcLst, paramLst):
      self.registerFunc(func, btnOpts=param)

    def onChange():
      self.firstRun = True
      self.clearFocusedRegion()
    win.mainImg.imgItem.sigImageChanged.connect(onChange)

    win.mainImg.registerDrawAction([CNST.DRAW_ACT_ADD, CNST.DRAW_ACT_REM], self._run_drawAct)
    win.mainImg.addTools(self.toolsEditor)
    self.vb: pg.ViewBox = win.mainImg.getViewBox()
    super().attachWinRef(win)

  def fillRegionMask(self):
    """Completely fill the focused region mask"""
    if self.mainImg.compSer is None: return
    filledImg = np.ones(self.mainImg.image.shape[:2], dtype='uint16')
    self.updateRegionFromMask(filledImg)

  @classmethod
  def clearProcessorHistory(cls):
    """
    Each time an update is made in the processor, it is saved so algorithmscan take
    past edits into account when performing their operations. Clearing that history
    will erase algorithm knowledge of past edits.
    """
    _historyMaskHolder[0].fill(0)

  def updateFocusedComp(self, newComp:pd.Series = None):
    self.updateFocusedPen_Fill()
    if self.mainImg.compSer[RTF.INST_ID] == -1:
      self.updateRegionFromDf(None)
      return
    oldId = self.mainImg.compSer[RTF.INST_ID]
    self.updateRegionFromDf(self.mainImg.compSer_asFrame)
    if newComp is None or oldId != newComp[RTF.INST_ID]:
      self.firstRun = True

  def updateFocusedPen_Fill(self):
    disp = self.win.compDisplay
    plt: MultiRegionPlot = disp.regionPlot
    mgrDf = self.win.compMgr.compDf
    regDf = plt.regionData.copy()
    focused = regDf[PRJ_ENUMS.FIELD_FOCUSED].to_numpy(bool)
    # Show IDs that were previously hidden
    for showId in regDf.index[regDf[RTF.VERTICES].apply(ComplexXYVertices.isEmpty)]:
      regDf.at[showId, RTF.VERTICES] = mgrDf.at[showId, RTF.VERTICES]

    # Make 0-area regions where the plot must be hidden
    for hideId in regDf.index[focused]:
      # Bug: Can't set array of values without indexer error since pandas thinks ComplexXYVertices is a 2d array
      regDf.at[hideId, RTF.VERTICES] = ComplexXYVertices()
    plt.resetRegionList(regDf, self.win.compDisplay.labelCol)

  def _run_drawAct(self, verts: XYVertices, param: PrjParam):
    # noinspection PyTypeChecker
    verts : XYVertices = verts.astype(int)
    if param == CNST.DRAW_ACT_ADD:
      self.run(fgVerts=verts)
    else:
      self.run(bgVerts=verts)

  def run(self, fgVerts: XYVertices=None, bgVerts: XYVertices=None):
    vertsDict = {}
    if fgVerts is not None:
      vertsDict['fgVerts'] = fgVerts
    if bgVerts is not None:
      vertsDict['bgVerts'] = bgVerts
    img = self.mainImg.image
    if img is None:
      compGrayscale = None
      compMask = None
    else:
      compGrayscale = self.region.toGrayImg(img.shape[:2])
      compMask = compGrayscale > 0
    # TODO: When multiple classes can be represented within focused image, this is where
    #  change will have to occur
    # Clip viewrange to min view axis area instead of max, which will happen internally
    # otherwise
    viewbox = self.mainImg.viewboxSquare()
    newGrayscale = self.curProcessor.run(
      image=img,
      prevCompMask=compMask,
      **vertsDict,
      firstRun=self.firstRun,
      viewbox=XYVertices(viewbox),
      prevCompVerts=ComplexXYVertices([r.stack() for r in self.region.regionData[RTF.VERTICES]])
    )
    if isinstance(newGrayscale, dict):
      newGrayscale = newGrayscale['image']
    newGrayscale = newGrayscale.astype('uint8')

    self.firstRun = False
    if not np.array_equal(newGrayscale, compGrayscale):
      self.updateRegionFromMask(newGrayscale)

  def updateRegionFromDf(self, newData: pd.DataFrame=None, offset: XYVertices=None):
    """
    Updates the current focused region using the new provided vertices
    :param newData: Dataframe to use.If *None*, the image will be totally reset and the component
      will be removed. Otherwise, the provided value will be used. For column information,
      see `makeMultiRegionDf`
    :param offset: Offset of newVerts relative to main image coordinates
    """
    fImg = self.mainImg
    # Since some calls to this function make an undo entry and others don't, be sure to save state for the undoable ones
    # revertId is only used when changedComp is true and an undo is valid
    newId = fImg.compSer[RTF.INST_ID]
    if newData is None or np.all(newData[RTF.VERTICES].apply(ComplexXYVertices.isEmpty)):
      newData = makeMultiRegionDf(0)

    self._maybeChangeFocusedComp(newData.index)

    if fImg.image is None:
      self.region.clear()
      self.regionBuffer.append(buffEntry(newId, ComplexXYVertices()))
      return

    oldData = self.region.regionData

    if offset is None:
      offset = XYVertices([[0,0]])
    # 0-center new vertices relative to FocusedImage image
    centeredData = newData
    if np.any(offset != 0):
      # Make a deep copy of verts (since they change) to preserve redos
      centeredData = centeredData.copy()
      centeredData[RTF.VERTICES] = self.applyOffset([copy.deepcopy(v) for v in newData[RTF.VERTICES]], offset)
    if oldData.equals(centeredData):
      return

    lblCol = self.win.compDisplay.labelCol
    self.region.resetRegionList(newRegionDf=centeredData, lblField=lblCol)
    self.region.focusById(centeredData.index)

    buffVerts = ComplexXYVertices()
    for inner in centeredData[RTF.VERTICES]: buffVerts.extend(inner)
    self.regionBuffer.append(buffEntry(newId, buffVerts))

  @staticmethod
  def applyOffset(complexVerts: Sequence[ComplexXYVertices], offset: XYVertices):
    centeredVerts = []
    for complexVerts in complexVerts:
      newVertList = ComplexXYVertices()
      for vertList in complexVerts:
        newVertList.append(vertList + offset)
      centeredVerts.append(newVertList)
    return centeredVerts

  def _maybeChangeFocusedComp(self, newIds: Sequence[int]):
    regionId = newIds[0] if len(newIds) else -1
    focusedId = self.mainImg.compSer[RTF.INST_ID]
    updated = regionId != focusedId
    if updated:
      if regionId in self.win.compMgr.compDf.index:
        self.win.compDisplay.selectRowsById([regionId])
      else:
        self.win.compDisplay.selectRowsById([])
    return updated

  def updateRegionFromMask(self, mask: BlackWhiteImg, offset=None, compId=None):
    if offset is None:
      offset = XYVertices([0,0])
    data = self.region.regionData.copy()
    if compId is None:
      compId = data.index[0] if len(data) else -1
    df = makeMultiRegionDf(vertices=[ComplexXYVertices.fromBwMask(mask)], idList=[compId])
    self.updateRegion_undoable(df, offset=offset)


  @DASM.undoable('Modify Focused Component')
  def updateRegion_undoable(self, newData: pd.DataFrame=None, offset: XYVertices=None):
    oldData = self.region.regionData.copy()
    self.updateRegionFromDf(newData, offset=offset)
    yield
    self.updateRegionFromDf(oldData)

  def acceptChanges(self, overrideVerts: ComplexXYVertices=None):
    # Add in offset from main image to VertexRegion vertices
    newVerts = overrideVerts or self.collapseRegionVerts()
    ser = self.mainImg.compSer
    ser[RTF.VERTICES] = newVerts
    self.updateFocusedComp()

  def collapseRegionVerts(self, simplify=True):
    """
    Region can consist of multiple separate complex vertices. However, the focused series
    can only contain one list of complex vertices. This function collapses all data in self.region
    into one list of complex vertices.

    :param simplify: Overlapping regions can be simplified by converting to and back from
      an image. This can be computationally intensive at times, in which case `simplify` can
      be set to *False*
    """
    outVerts = ComplexXYVertices([verts for cplxVerts in self.region.regionData[RTF.VERTICES] for verts in cplxVerts])
    if simplify:
      outVerts = ComplexXYVertices.fromBwMask(outVerts.toMask())
    return outVerts

  def clearFocusedRegion(self):
    # Reset drawn comp vertices to nothing
    # Only perform action if image currently exists
    if self.mainImg.compSer is None:
      return
    self.updateRegion_undoable(None)

  def resetFocusedRegion(self):
    """Reset the focused image by restoring the region mask to the last saved state"""
    if self.mainImg.compSer is None:
      return
    self.updateRegion_undoable(self.mainImg.compSer_asFrame)

  def _onActivate(self):
    self.region.show()
    self.updateFocusedPen_Fill()

  def _onDeactivate(self):
    self.region.hide()
    self.win.compDisplay.regionPlot.updateColors()

  def getRegionHistory(self):
    outImgs = []
    if not self.regionBuffer:
      return None, []
    firstId = self.regionBuffer[-1].id_
    bufferRegions = [buf.vertices for buf in self.regionBuffer if buf.id_ == firstId]

    if not bufferRegions:
      return None, []

    # First find offset and img size so we don't
    # have to keep copying a full image sized output every time
    allVerts = np.vstack([v.stack() for v in bufferRegions])
    initialImg, slices = getCroppedImg(self.mainImg.image, allVerts)
    imShape = initialImg.shape[:2]
    offset = slices[0]
    img = np.zeros(imShape, bool)
    outImgs.append(img)
    for singleRegionVerts in bufferRegions:
      # Copy to avoid screwing up undo buffer!
      copied = ComplexXYVertices([subV - offset for subV in singleRegionVerts])
      img = copied.toMask(imShape, warnIfTooSmall=False)
      outImgs.append(img)
    return initialImg, outImgs


  def playbackRegionHistory(self):
    initialImg, history = self.getRegionHistory()
    if initialImg is None:
      warn('No edits found, nothing to do', UserWarning)
      return
    # Add current state as final result
    history += [history[-1]]
    diffs = [showMaskDiff(o, n) for (o, n) in zip(history, history[1:])]
    self.playbackWindow.setDiffs(diffs)
    self.playbackWindow.displayPlt.setImage(initialImg)
    self.playbackWindow.show()
    self.playbackWindow.raise_()