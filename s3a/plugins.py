from functools import partial
from functools import partial
from typing import Optional

import cv2 as cv
import numpy as np
import pandas as pd
from pyqtgraph.Qt import QtWidgets

from s3a import FR_SINGLETON, FR_CONSTS as FRC, REQD_TBL_FIELDS as RTF, FRComplexVertices, \
  FRVertices, FRParam
from s3a.generalutils import frPascalCaseToTitle
from s3a.models.s3abase import S3ABase
from s3a.parameditors import FRParamEditorDockGrouping
from s3a.parameditors.genericeditor import FRTableFieldPlugin
from s3a.processing import FRImageProcess
from s3a.processing.algorithms import _historyMaskHolder
from s3a.structures import NChanImg, FRAlgProcessorError
from s3a.views.regions import FRVertexDefinedImg


class FREEHAND_OPTS:
  ROI_SZ = FRParam('ROI Size (px)', 5, helpText='Area filled by the draw tool')

class FRVerticesPlugin(FRTableFieldPlugin):
  name = 'Vertices'

  @classmethod
  def __initEditorParams__(cls):
    super().__initEditorParams__()
    cls.procCollection = FR_SINGLETON.imgProcClctn.createProcessorForClass(cls)
    (cls.resetRegionAct, cls.fillRegionAct,
     cls.clearRegionAct, cls.clearHistoryAct) = cls.toolsEditor.registerProps(
      cls, [FRC.TOOL_RESET_FOC_REGION, FRC.TOOL_FILL_FOC_REGION,
            FRC.TOOL_CLEAR_FOC_REGION, FRC.TOOL_CLEAR_HISTORY],
      asProperty=False, ownerObj=cls)

    dockGroup = FRParamEditorDockGrouping([cls.toolsEditor, cls.procCollection],
                                          frPascalCaseToTitle(cls.name))
    cls.docks = dockGroup

  def __init__(self):
    self._addFreehandProcessor()

    self.region = FRVertexDefinedImg()
    self.region.hide()
    self.firstRun = True

    # Disable local cropping on primitive grab cut by default
    self.procCollection.nameToProcMapping['Primitive Grab Cut'].setStageEnabled(['Crop To Local Area'], False)

  def attachS3aRef(self, s3a: S3ABase):
    super().attachS3aRef(s3a)
    s3a.focusedImg.addItem(self.region)

    self.clearRegionAct.sigActivated.connect(lambda: self.updateRegionFromVerts(None))
    def fillAct():
      if self.focusedImg.image is None: return
      filled = np.ones(self.focusedImg.image.shape[:2], bool)
      self.region.updateFromMask(filled)
    self.fillRegionAct.sigActivated.connect(fillAct)
    self.resetRegionAct.sigActivated.connect(
      lambda: self.updateRegionFromVerts(self.focusedImg.compSer[RTF.VERTICES]))
    self.clearHistoryAct.sigActivated.connect(
      lambda: _historyMaskHolder[0].fill(0)
    )
    self.resetRegionAct.sigActivated.connect(self.resetFocusedRegion)

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

  def _addFreehandProcessor(self):
    def freehand(image: NChanImg, fgVerts: FRVertices, penSize=1, penShape='circle'):
      self.focusedImg.shapeCollection.curShape.handleSize = penSize

      out = np.zeros(image.shape[:2], dtype='uint8')
      drawFns = {
        'circle': lambda pt: cv.circle(out, tuple(pt), penSize//2, 1, -1),
        'rectangle': lambda pt: cv.rectangle(out, tuple(pt-penSize//2), tuple(pt+penSize//2), 1, -1)
      }
      try:
        drawFn = drawFns[penShape]
      except KeyError:
        raise FRAlgProcessorError(f"Can't understand shape {penShape}. Must be one of:\n"
                                  f"{','.join(drawFns)}")
      if len(fgVerts) > 1:
        FRComplexVertices([fgVerts]).toMask(out, 1, False, warnIfTooSmall=False)
      else:
        for vert in fgVerts:
          drawFn(vert)
      return out > 0

    def makeProc():
      proc = FRImageProcess.fromFunction(freehand, needsWrap=True)
      proc.excludedStages = [['Basic Region Operations']]
      return proc

    FR_SINGLETON.imgProcClctn.addProcessCtor(makeProc)


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
    if self.region.verts != centeredVerts:
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
    # Reset drawn comp vertices to nothing
    # Only perform action if image currently exists
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