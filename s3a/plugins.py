from functools import partial
from functools import partial
from typing import Optional

import numpy as np
import pandas as pd
from pyqtgraph.Qt import QtWidgets

from s3a import FR_SINGLETON, FR_CONSTS as FRC, REQD_TBL_FIELDS as RTF, FRComplexVertices, \
  FRVertices, FRParam, FRComponentIO as frio
from s3a.generalutils import frPascalCaseToTitle
from s3a.models.s3abase import S3ABase
from s3a.parameditors import FRParamEditorDockGrouping
from s3a.parameditors.genericeditor import FRTableFieldPlugin
from s3a.processing.algorithms import _historyMaskHolder
from s3a.structures import NChanImg, GrayImg
from s3a.views.regions import FRMultiRegionPlot, makeMultiRegionDf


class FRVerticesPlugin(FRTableFieldPlugin):
  name = 'Vertices'

  @classmethod
  def __initEditorParams__(cls):
    super().__initEditorParams__()
    cls.procCollection = FR_SINGLETON.imgProcClctn.createProcessorForClass(cls)

    dockGroup = FRParamEditorDockGrouping([cls.toolsEditor, cls.procCollection],
                                          frPascalCaseToTitle(cls.name))
    cls.docks = dockGroup

  def __init__(self):
    self.region = FRMultiRegionPlot()
    self.region.hide()
    self.firstRun = True

    # Disable local cropping on primitive grab cut by default
    self.procCollection.nameToProcMapping['Primitive Grab Cut'].setStageEnabled(['Crop To Local Area'], False)

  def attachS3aRef(self, s3a: S3ABase):
    super().attachS3aRef(s3a)
    s3a.focusedImg.addItem(self.region)

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
      self.toolsEditor.registerFunc(func, btnOpts=param)


  def updateAll(self, mainImg: Optional[NChanImg], newComp: Optional[pd.Series] = None):
    if self.focusedImg.image is None:
      self.updateRegionFromDf(None)
      return
    self.updateRegionFromDf(self.focusedImg.compSer.to_frame().T, self.focusedImg.bbox[0, :])
    self.firstRun = True

  def handleShapeFinished(self, roiVerts: FRVertices):
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
  def updateRegionFromDf(self, newData: pd.DataFrame=None, offset: FRVertices=None):
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
    # 0-center new vertices relative to FRFocusedImage image
    # Make a copy of each list first so we aren't modifying the
    # original data
    centeredData = newData.copy()
    centeredVerts = []
    for complexVerts in centeredData[RTF.VERTICES]:
      newVertList = FRComplexVertices()
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
    self.updateRegionFromDf(df, offset=FRVertices([0, 0]))
    pass

  def acceptChanges(self, overrideVerts: FRComplexVertices=None):
    # Add in offset from main image to FRVertexRegion vertices
    ser = self.focusedImg.compSer
    if overrideVerts is not None:
      ser.loc[RTF.VERTICES] = overrideVerts
      return
    newVerts_lst = self.region.regionData[RTF.VERTICES].copy()
    newVerts = FRComplexVertices()
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

class Dummy(FRTableFieldPlugin):
  name = 'Dummy'

  @classmethod
  def __initEditorParams__(cls):
    super().__initEditorParams__()
    ps = [FRParam(l, value=f'Ctrl+{l}', pType='registeredaction') for l in 'abcd']
    def alert(btnName):
      QtWidgets.QMessageBox.information(cls.toolsEditor, 'Button clicked',
                                        f'{btnName} button clicked!')

    for p in ps:
      prop = cls.toolsEditor.registerProp(cls, p, asProperty=False, ownerObj=cls)
      prop.sigActivated.connect(partial(alert, p.name))

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)


  def updateAll(self, mainImg: Optional[NChanImg], newComp: Optional[pd.Series] = None):
    pass

  def handleShapeFinished(self, roiVerts: FRVertices):
    pass

  def acceptChanges(self):
    pass