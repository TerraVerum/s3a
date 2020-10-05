from functools import partial
from functools import partial
from typing import Optional

import numpy as np
import pandas as pd
from pyqtgraph.Qt import QtWidgets

from s3a import FR_SINGLETON, FR_CONSTS as FRC, REQD_TBL_FIELDS as RTF, FRComplexVertices, \
  FRVertices, FRParam
from s3a.generalutils import frPascalCaseToTitle, dynamicDocstring, frParamToPgParamDict
from s3a.models.s3abase import S3ABase
from s3a.parameditors import FRParamEditorDockGrouping
from s3a.parameditors.genericeditor import FRTableFieldPlugin
from s3a.processing.algorithms import _historyMaskHolder
from s3a.structures import NChanImg
from s3a.views.regions import FRVertexDefinedImg


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
    self.region = FRVertexDefinedImg()
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
      filled = np.ones(self.focusedImg.image.shape[:2], bool)
      self.region.updateFromMask(filled)
    def clear():
      """
      Clear the vertices in the focused image
      """
      self.updateRegionFromVerts(None)
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
      self.updateRegionFromVerts(None)
      return
    self.updateRegionFromVerts(self.focusedImg.compSer[RTF.VERTICES], self.focusedImg.bbox[0,:])
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

    compMask = None if img is None else self.region.embedMaskInImg(img.shape[:2])
    newMask = self.curProcessor.run(image=img, prevCompMask=compMask, **vertsDict,
                                    firstRun=self.firstRun)
    self.firstRun = False
    if not np.array_equal(newMask,compMask):
      self.region.updateFromMask(newMask)


  @FR_SINGLETON.actionStack.undoable('Modify Focused Component')
  def updateRegionFromVerts(self, newVerts: FRComplexVertices=None, offset: FRVertices=None):
    """
    Updates the current focused region using the new provided vertices
    :param newVerts: Verts to use.If *None*, the image will be totally reset and the component
      will be removed. Otherwise, the provided value will be used.
    :param offset: Offset of newVerts relative to main image coordinates
    """
    fImg = self.focusedImg
    if fImg.image is None:
      self.region.clear()
      return
    oldVerts = self.region.verts
    oldRegionImg = self.region.image

    oldSelfImg = fImg.image
    oldSer = fImg.compSer

    if offset is None:
      offset = fImg.bbox[0,:]
    if newVerts is None:
      newVerts = FRComplexVertices()
    # 0-center new vertices relative to FRFocusedImage image
    # Make a copy of each list first so we aren't modifying the
    # original data
    centeredVerts = newVerts.copy()
    for vertList in centeredVerts:
      vertList -= offset
    if np.any(fImg.bbox[:,0] != offset) or self.region.verts != centeredVerts:
      self.region.updateFromVertices(centeredVerts)
      yield
    else:
      return
    if (fImg.compSer.loc[RTF.INST_ID] != oldSer.loc[RTF.INST_ID]
        or fImg.image is None):
      fImg.updateAll(oldSelfImg, oldSer)
    self.region.updateFromVertices(oldVerts, oldRegionImg)

  def acceptChanges(self, overrideVerts: FRComplexVertices=None):
    # Add in offset from main image to FRVertexRegion vertices
    if overrideVerts is not None:
      self.focusedImg.compSer.loc[RTF.VERTICES] = overrideVerts
      return
    newVerts = self.region.verts.copy()
    for vertList in newVerts:
      vertList += self.focusedImg.bbox[0,:]
    self.focusedImg.compSer.at[RTF.VERTICES] = newVerts

  def clearFocusedRegion(self):
    # Reset drawn comp vertices to nothing
    # Only perform action if image currently exists
    if self.focusedImg.image is None:
      return
    self.updateRegionFromVerts(None)

  def resetFocusedRegion(self):
    """Reset the focused image by restoring the region mask to the last saved state"""
    if self.focusedImg.image is None:
      return
    self.updateRegionFromVerts(self.focusedImg.compSer[RTF.VERTICES])

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