from typing import Optional, Sequence

import numpy as np
import pandas as pd

from s3a import FR_SINGLETON, FR_CONSTS as FRC, XYVertices, REQD_TBL_FIELDS as RTF, \
  ComplexXYVertices, ComponentIO as frio, models
from s3a.models.s3abase import S3ABase
from s3a.processing.algorithms import _historyMaskHolder
from s3a.structures import NChanImg, GrayImg
from s3a.views.regions import MultiRegionPlot, makeMultiRegionDf
from .base import TableFieldPlugin
from ..constants import FR_ENUMS
import numpy as np

from ..processing.processing import GlobalPredictionProcess


class VerticesPlugin(TableFieldPlugin):
  name = 'Vertices'

  @classmethod
  def __initEditorParams__(cls):
    super().__initEditorParams__()
    cls.procCollection = FR_SINGLETON.imgProcClctn.createProcessorForClass(cls)

    cls.dock.addEditors([cls.procCollection])

  def __init__(self):
    super().__init__()
    self.region = MultiRegionPlot()
    self.region.hide()
    self.firstRun = True

    # Disable local cropping on primitive grab cut by default
    self.procCollection.nameToProcMapping['Primitive Grab Cut'].setStageEnabled(['Crop To Local Area'], False)

  def attachWinRef(self, win: S3ABase):
    win.focusedImg.addItem(self.region)

    def fill():
      """Completely fill the focused region mask"""
      if self.focusedImg.image is None: return
      clsIdx = self.focusedImg.classIdx
      filledImg = np.ones(self.focusedImg.image.shape[:2], dtype='uint16')*(clsIdx+1)
      self.updateRegionFromClsImg(filledImg)
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
    paramLst = [FRC.TOOL_RESET_FOC_REGION, FRC.TOOL_FILL_FOC_REGION,
                FRC.TOOL_CLEAR_FOC_REGION, FRC.TOOL_CLEAR_HISTORY]
    for func, param in zip(funcLst, paramLst):
      param.opts['ownerObj'] = win.focusedImg
      self.registerFunc(func, btnOpts=param)

    super().attachWinRef(win)



  def updateAll(self, mainImg: Optional[NChanImg], newComp: Optional[pd.Series] = None):
    if self.focusedImg.image is None:
      self.updateRegionFromDf(None)
      return
    self.updateRegionFromDf(self.focusedImg.compSer.to_frame().T, self.focusedImg.bbox[0, :])
    self.firstRun = True

  def handleShapeFinished(self, roiVerts: XYVertices):
    roiVerts = roiVerts.astype(int)
    vertsDict = {}
    act = self.focusedImg.drawAction
    img = self.focusedImg.image

    if act == FRC.DRAW_ACT_ADD:
      vertsDict['fgVerts'] = roiVerts
    elif act == FRC.DRAW_ACT_REM:
      vertsDict['bgVerts'] = roiVerts

    if img is None:
      compGrayscale = None
      compMask = None
    else:
      compGrayscale = self.region.toGrayImg(img.shape[:2])
      compMask = compGrayscale > 0
    # TODO: When multiple classes can be represented within focused image, this is where
    #  change will have to occur
    newGrayscale = self.curProcessor.run(image=img, prevCompMask=compMask, **vertsDict,
                                    firstRun=self.firstRun).astype('uint16')
    newGrayscale *= (self.focusedImg.classIdx+1)
    self.firstRun = False
    if not np.array_equal(newGrayscale, compGrayscale):
      self.updateRegionFromClsImg(newGrayscale)


  @FR_SINGLETON.actionStack.undoable('Modify Focused Component')
  def updateRegionFromDf(self, newData: pd.DataFrame=None, offset: XYVertices=None):
    """
    Updates the current focused region using the new provided vertices
    :param newData: Dataframe to use.If *None*, the image will be totally reset and the component
      will be removed. Otherwise, the provided value will be used. For column information,
      see `makeMultiRegionDf`
    :param offset: Offset of newVerts relative to main image coordinates
    """
    fImg = self.focusedImg
    if fImg.image is None:
      self.region.clear()
      return
    oldData = self.region.regionData

    oldSelfImg = fImg.image
    oldSer = fImg.compSer

    if offset is None:
      offset = fImg.bbox[0,:]
    if newData is None:
      newData = makeMultiRegionDf(0)
    # 0-center new vertices relative to FocusedImage image
    # Make a copy of each list first so we aren't modifying the
    # original data
    centeredData = newData.copy()
    centeredVerts = []
    for complexVerts in centeredData[RTF.VERTICES]:
      newVertList = ComplexXYVertices()
      for vertList in complexVerts:
        newVertList.append(vertList-offset)
      centeredVerts.append(newVertList)
    centeredData[RTF.VERTICES] = centeredVerts
    if np.any(fImg.bbox[0,:] != offset) or not oldData.equals(centeredData):
      self.region.resetRegionList(newRegionDf=centeredData)
      yield
    else:
      return
    if (fImg.compSer.loc[RTF.INST_ID] != oldSer.loc[RTF.INST_ID]
        or fImg.image is None):
      fImg.updateAll(oldSelfImg, oldSer)
    self.region.resetRegionList(oldData, convertClasses=False)

  def updateRegionFromClsImg(self, clsImg: GrayImg):
    df = frio.buildFromClassPng(clsImg)
    self.updateRegionFromDf(df, offset=XYVertices([0, 0]))
    pass

  def acceptChanges(self, overrideVerts: ComplexXYVertices=None):
    # Add in offset from main image to VertexRegion vertices
    ser = self.focusedImg.compSer
    if overrideVerts is not None:
      ser.loc[RTF.VERTICES] = overrideVerts
      return
    newVerts_lst = self.region.regionData[RTF.VERTICES].copy()
    newVerts = ComplexXYVertices()
    for verts in newVerts_lst:
      newVerts.extend(verts.copy())
    for vertList in newVerts:
      vertList += self.focusedImg.bbox[0,:]
    ser.at[RTF.VERTICES] = newVerts

  def clearFocusedRegion(self):
    # Reset drawn comp vertices to nothing
    # Only perform action if image currently exists
    if self.focusedImg.image is None:
      return
    self.updateRegionFromDf(None)

  def resetFocusedRegion(self):
    """Reset the focused image by restoring the region mask to the last saved state"""
    if self.focusedImg.image is None:
      return
    self.updateRegionFromDf(self.focusedImg.compSer.to_frame().T)

  def _onActivate(self):
    self.region.show()

  def _onDeactivate(self):
    self.region.hide()


class GlobalPredictionsPlugin(TableFieldPlugin):
  name = 'Global Predictions'

  mgr: models.tablemodel.ComponentMgr

  @classmethod
  def __initEditorParams__(cls):
    super().__initEditorParams__()
    cls.procCollection = FR_SINGLETON.globalPredClctn.createProcessorForClass(cls)
    cls.dock.addEditors([cls.procCollection])

  def __init__(self):
    super().__init__()
    self.origToDupMappingDf = pd.DataFrame(columns=['orig', 'duplicate'])

  def attachWinRef(self, win: models.s3abase.S3ABase):
    super().attachWinRef(win)
    def onChange(changeDict):
      origIds = self.origToDupMappingDf['orig']
      keeps = ~np.isin(origIds, changeDict['deleted'])
      self.origToDupMappingDf = self.origToDupMappingDf.loc[keeps]
    win.compMgr.sigCompsChanged.connect(onChange)
    self.mgr = win.compMgr

  def handleShapeFinished(self, roiVerts: XYVertices):
    pass

  def acceptChanges(self):
    origVerts = self.focusedImg.compSer[RTF.VERTICES].copy()
    origOffset = origVerts.stack().min(0)
    # Account for margin in focused image
    for verts in origVerts:
      verts -= origOffset
    origId = self.focusedImg.compSer[RTF.INST_ID]
    df = self.mgr.compDf
    modifiedCompIds = self.origToDupMappingDf.loc[self.origToDupMappingDf['orig'] == origId, 'duplicate']
    toModifyIdxs = df.index.isin(modifiedCompIds)
    toModifyVerts = df.loc[toModifyIdxs, RTF.VERTICES]
    allNewVerts = []
    for complexVerts in toModifyVerts: # type: ComplexXYVertices
      offset = complexVerts.stack().min(0)
      newVerts = []
      for verts in origVerts:
        newVerts.append(verts + offset)
      allNewVerts.append(ComplexXYVertices(newVerts))
    toAddDf = pd.DataFrame(np.c_[toModifyIdxs, allNewVerts], columns=[RTF.INST_ID, RTF.VERTICES])
    toAddDf = toAddDf.set_index(RTF.INST_ID, drop=False)
    self.mgr.addComps(toAddDf, FR_ENUMS.COMP_ADD_AS_MERGE)


  def updateAll(self, mainImg: Optional[NChanImg], newComp: Optional[pd.Series] = None):
    if self.focusedImg.image is None:
      return
    boxes: Sequence[XYVertices] = self.curProcessor.run(image=self.focusedImg.image, globalImage=self.win.mainImg.image)
    verts = [ComplexXYVertices([box]) for box in boxes]
    newComps = FR_SINGLETON.tableData.makeCompDf(len(boxes))
    newComps[RTF.VERTICES] = verts
    changeList = self.win.compMgr.addComps(newComps)['added']
    origId = np.tile(self.focusedImg.compSer[RTF.INST_ID], len(boxes))
    newData = np.c_[origId, changeList]
    mappingDf = pd.DataFrame(newData, columns=self.origToDupMappingDf.columns)
    self.origToDupMappingDf: pd.DataFrame = self.origToDupMappingDf.append(mappingDf).reset_index(drop=True)


def dummyProc(image: np.ndarray, globalImage: np.ndarray, nboxes=5):
  imShape = np.array(globalImage.shape[:2])

  # Reverse shape for x-y instead of row-col
  box = imShape[::-1]*np.array([[0,0], [0,1], [1,1], [1,0]])
  boxes = box*np.linspace(.1, .9, nboxes)[:,None, None]
  boxes = list(boxes.astype(int))
  return boxes

FR_SINGLETON.globalPredClctn.addProcessFunction(dummyProc, GlobalPredictionProcess)