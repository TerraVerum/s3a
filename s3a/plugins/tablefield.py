from warnings import warn

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from utilitys import PrjParam
from utilitys.widgets import ImgViewer

from s3a import PRJ_SINGLETON, PRJ_CONSTS as CNST, XYVertices, REQD_TBL_FIELDS as RTF, \
  ComplexXYVertices
from s3a.models.s3abase import S3ABase
from s3a.processing.algorithms import _historyMaskHolder
from s3a.structures import BlackWhiteImg
from s3a.views.regions import MultiRegionPlot, makeMultiRegionDf
from .base import TableFieldPlugin
from ..constants import PRJ_ENUMS
from ..generalutils import getCroppedImg, showMaskDiff


class VerticesPlugin(TableFieldPlugin):
  name = 'Vertices'

  @classmethod
  def __initEditorParams__(cls):
    super().__initEditorParams__()
    cls.procCollection = PRJ_SINGLETON.imgProcClctn.createProcessorForClass(cls, cls.name + ' Processor')

    cls.dock.addEditors([cls.procCollection])

  def __init__(self):
    super().__init__()
    self.region = MultiRegionPlot()
    self.region.hide()
    self.firstRun = True
    self.playbackWindow = ImgViewer()
    ci = self.playbackWindow.changingItem = pg.ImageItem()
    self.playbackWindow.addItem(ci)
    ci.setOpacity(0.5)


  def attachWinRef(self, win: S3ABase):
    win.mainImg.addItem(self.region)

    def fill():
      """Completely fill the focused region mask"""
      if self.mainImg.compSer is None: return
      filledImg = np.ones(self.mainImg.image.shape[:2], dtype='uint16')
      self.updateRegionFromMask(filledImg)
    def clear():
      """
      Clear the vertices in the focused image
      """
      self.updateRegionFromDf(None)
    def clearProcessorHistory():
      """
      Each time an update is made in the processor, it is saved so algorithmscan take
      past edits into account when performing their operations. Clearing that history
      will erase algorithm knowledge of past edits.
      """
      _historyMaskHolder[0].fill(0)

    funcLst = [self.resetFocusedRegion, fill, clear, clearProcessorHistory]
    paramLst = [CNST.TOOL_RESET_FOC_REGION, CNST.TOOL_FILL_FOC_REGION,
                CNST.TOOL_CLEAR_FOC_REGION, CNST.TOOL_CLEAR_HISTORY]
    for func, param in zip(funcLst, paramLst):
      self.registerFunc(func, btnOpts=param)

    self.registerFunc(self.playbackRegionHistory)

    win.mainImg.registerDrawAction([CNST.DRAW_ACT_ADD, CNST.DRAW_ACT_REM], self._run_drawAct)
    win.mainImg.addTools(self.toolsEditor)
    self.vb: pg.ViewBox = win.mainImg.getViewBox()
    super().attachWinRef(win)

  def updateFocusedComp(self, newComp:pd.Series = None):
    if self.mainImg.compSer[RTF.INST_ID] == -1:
      self.updateRegionFromDf(None)
      return
    oldId = self.mainImg.compSer[RTF.INST_ID]
    self.updateRegionFromDf(self.mainImg.compSer_asFrame)
    self.clearFocusedPen_Fill()
    if newComp is None or oldId != newComp[RTF.INST_ID]:
      self.firstRun = True

  def clearFocusedPen_Fill(self):
    plt: MultiRegionPlot = self.win.compDisplay.regionPlot
    ids = plt.regionData[PRJ_ENUMS.FIELD_FOCUSED].to_numpy()
    for name, fn in zip(['pen', 'brush'], [pg.mkPen, pg.mkBrush]):
      clrs = plt.data[name]
      clrs[ids] = fn('0000')
      plt.data[name] = clrs
    plt.updateSpots(plt.data)

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
      prevCompVerts=self.mainImg.compSer[RTF.VERTICES]
    ).astype('uint8')

    self.firstRun = False
    if not np.array_equal(newGrayscale, compGrayscale):
      self.updateRegionFromMask(newGrayscale)

  @PRJ_SINGLETON.actionStack.undoable('Modify Focused Component')
  def updateRegionFromDf(self, newData: pd.DataFrame=None, offset: XYVertices=None):
    """
    Updates the current focused region using the new provided vertices
    :param newData: Dataframe to use.If *None*, the image will be totally reset and the component
      will be removed. Otherwise, the provided value will be used. For column information,
      see `makeMultiRegionDf`
    :param offset: Offset of newVerts relative to main image coordinates
    """
    fImg = self.mainImg
    if newData is None or np.all(newData[RTF.VERTICES].apply(ComplexXYVertices.isEmpty)):
      newData = makeMultiRegionDf(0)
    if fImg.image is None:
      self.region.clear()
      return
    oldData = self.region.regionData

    oldSer = fImg.compSer

    if offset is None:
      offset = XYVertices([[0,0]])
    # 0-center new vertices relative to FocusedImage image
    # Make a copy of each list first so we aren't modifying the
    # original data
    centeredData = newData.copy()
    if np.any(offset != 0):
      centeredVerts = []
      for complexVerts in centeredData[RTF.VERTICES]:
        newVertList = ComplexXYVertices()
        for vertList in complexVerts:
          newVertList.append(vertList+offset)
        centeredVerts.append(newVertList)
      centeredData[RTF.VERTICES] = centeredVerts

    if oldData.equals(centeredData):
      return
    else:
      self.region.resetRegionList(newRegionDf=centeredData)
      self.region.focusById(centeredData.index)
      yield
    if (fImg.compSer.loc[RTF.INST_ID] != oldSer.loc[RTF.INST_ID]
        or fImg.image is None):
      self.win.changeFocusedComp(oldSer)
    self.region.resetRegionList(oldData)

  def updateRegionFromMask(self, mask: BlackWhiteImg, offset=None):
    if offset is None:
      offset = XYVertices([0,0])
    df = makeMultiRegionDf(vertices=[ComplexXYVertices.fromBwMask(mask)])
    self.updateRegionFromDf(df, offset=offset)

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
    self.updateRegionFromDf(None)

  def resetFocusedRegion(self):
    """Reset the focused image by restoring the region mask to the last saved state"""
    if self.mainImg.compSer is None:
      return
    self.updateRegionFromDf(self.mainImg.compSer_asFrame)

  def _onActivate(self):
    self.region.show()
    self.clearFocusedPen_Fill()

  def _onDeactivate(self):
    self.region.hide()
    self.win.compDisplay.regionPlot.updateColors()

  def getRegionHistory(self):
    outImgs = []
    stack = PRJ_SINGLETON.actionStack.actions
    bufferRegions = []
    for act in stack:
      if act.descr == 'Change Focused Component':
        # Signals a new chain of events, clear out the old
        bufferRegions.clear()
        continue
      elif act.descr != 'Modify Focused Component':
        continue
      elif act.args[1] is None or len(act.args[1]) == 0:
        # "None" occurrences  denote empty regions
        bufferRegions.append(ComplexXYVertices())
        continue
      # newData = arg 1
      regData = act.args[1].iloc[0]

      bufferRegions.append(regData[RTF.VERTICES])

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
    diffs = [showMaskDiff(o, n) for (o, n) in zip(history, history[1:])]
    # Add current state as final result
    diffs.append(np.tile(history[-1].astype('uint8')[...,None]*255, (1,1,3)))
    if initialImg is None:
      warn('No edits found, nothing to do', UserWarning)
    self.playbackWindow.setImage(initialImg)
    changingItem = self.playbackWindow.changingItem
    changingItem.clear()
    self.playbackWindow.show()
    # Add reference to avoid gc
    ii = 0
    def update():
      nonlocal ii
      changingItem.setImage(diffs[ii])
      ii += 1
      if ii == len(diffs):
        tim.stop()
        tim.deleteLater()
    tim = QtCore.QTimer()
    tim.timeout.connect(update)
    tim.start(500)