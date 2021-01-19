import numpy as np
import pandas as pd
import pyqtgraph as pg

from s3a import FR_SINGLETON, PRJ_CONSTS as CNST, XYVertices, REQD_TBL_FIELDS as RTF, \
  ComplexXYVertices
from s3a.models.s3abase import S3ABase
from s3a.processing.algorithms import _historyMaskHolder
from s3a.structures import BlackWhiteImg
from utilitys import PrjParam
from s3a.views.regions import MultiRegionPlot, makeMultiRegionDf
from .base import TableFieldPlugin
from ..constants import PRJ_ENUMS

class VerticesPlugin(TableFieldPlugin):
  name = 'Vertices'

  @classmethod
  def __initEditorParams__(cls):
    super().__initEditorParams__()
    cls.procCollection = FR_SINGLETON.imgProcClctn.createProcessorForClass(cls, cls.name + ' Processor')

    cls.dock.addEditors([cls.procCollection])

  def __init__(self):
    super().__init__()
    self.region = MultiRegionPlot()
    self.region.hide()
    self.firstRun = True

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

    win.mainImg.registerDrawAction([CNST.DRAW_ACT_ADD, CNST.DRAW_ACT_REM], self._run_drawAct)
    win.mainImg.addTools(self.toolsEditor)
    self.vb: pg.ViewBox = win.mainImg.getViewBox()
    super().attachWinRef(win)

  def updateFocusedComp(self, newComp:pd.Series = None):
    if self.mainImg.compSer[RTF.INST_ID] == -1:
      self.updateRegionFromDf(None)
      return
    self.updateRegionFromDf(self.mainImg.compSer_asFrame)
    self.clearFocusedPen_Fill()
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
      self.firstRun = True
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
    vbRange = np.array(self.vb.viewRange())
    span = np.diff(vbRange).flatten()
    center = vbRange[:,0]+span/2
    minSpan = np.min(span)
    offset = center - minSpan/2
    viewbox = minSpan*np.array([[0,0], [0,1], [1,1], [1,0]]) + offset
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

  @FR_SINGLETON.actionStack.undoable('Modify Focused Component')
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