from dataclasses import dataclass

import cv2 as cv
import numpy as np
from skimage import morphology

from Annotator.generalutils import splitListAtNans, nanConcatList
from Annotator.params import FRParamGroup, FRParam, newParam
from Annotator.processing import growSeedpoint
from .FRGraphics.parameditors import FR_SINGLETON
from .generalutils import getClippedBbox
from Annotator.interfaces import FRImageProcessor
from .params import FRVertices
from .processing import getVertsFromBwComps, getBwComps


@dataclass
class _FRDefaultAlgImpls(FRParamGroup):
  CLS_REGION_GROW : FRParam = newParam('Region Growing')

  PROP_SEED_THRESH: FRParam = newParam('Seedpoint Threshold in Main Image', 10.)
  PROP_MIN_COMP_SZ: FRParam = newParam('Minimum New Component Size (px)', 50)
  PROP_NEW_COMP_SZ: FRParam = newParam('New Component Side Length (px)', 30)
  PROP_MARGIN     : FRParam = newParam('New Component Margin (px)', 5)

IMPLS = _FRDefaultAlgImpls()

@FR_SINGLETON.algParamMgr_.registerClass(IMPLS.CLS_REGION_GROW)
class RegionGrow(FRImageProcessor):
  @FR_SINGLETON.algParamMgr_.registerProp(IMPLS.PROP_NEW_COMP_SZ)
  def newCompSz(self): pass
  @FR_SINGLETON.algParamMgr_.registerProp(IMPLS.PROP_MIN_COMP_SZ)
  def minCompSz(self): pass
  @FR_SINGLETON.algParamMgr_.registerProp(IMPLS.PROP_MARGIN)
  def margin(self): pass
  @FR_SINGLETON.algParamMgr_.registerProp(IMPLS.PROP_SEED_THRESH)
  def seedThresh(self): pass

  def localCompEstimate(self, prevCompMask: np.ndarray, fgVerts: FRVertices = None,
                        bgVerts: FRVertices = None) -> np.ndarray:
    # TODO: Make this code more robust
    needsInvert = False
    if fgVerts is None:
      # If given background points, invert the component and grow the background
      fgVerts = bgVerts
      needsInvert = True
    fgVerts[fgVerts < 0] = 0
    shape = prevCompMask.shape[0:2]
    for idx in range(2):
      fgVerts[fgVerts[:, idx] > shape[idx], idx] = shape[idx]
    croppedImg, cropOffset = self.getCroppedImg(fgVerts, self.margin)
    if croppedImg.size == 0:
      return prevCompMask
    centeredFgVerts = fgVerts - cropOffset[0:2]

    # For small enough shapes, get all boundary pixels instead of just shape vertices
    if fgVerts.connected and np.prod(croppedImg.shape[0:2]) < 50e3:
      # Use all vertex points, not just the defined corners
      fillPolyArg = splitListAtNans(centeredFgVerts)
      tmpImg = np.zeros(croppedImg.shape[0:2], dtype='uint8')
      tmpBwShape = cv.fillPoly(tmpImg, fillPolyArg, 1)
      contours, _ = cv.findContours(tmpBwShape, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
      if len(contours) == 0:
        return prevCompMask
      centeredFgVerts = FRVertices(nanConcatList([contours[0][:,0,:]]))
    # TODO: Find a better method of determining whether to use all bounds
    newRegion = growSeedpoint(croppedImg, centeredFgVerts, self.seedThresh, self.minCompSz)
    newRegion = morphology.opening(newRegion, morphology.square(3))

    if needsInvert:
      newRegion = np.invert(newRegion)
    prevCompMask[cropOffset[1]:cropOffset[3], cropOffset[0]:cropOffset[2]] |= newRegion

      # Remember to account for the vertex offset
    newVerts = getVertsFromBwComps(prevCompMask)
    #for vertList in newVerts:
      #vertList += cropOffset[0:2]
    newVerts = [FRVertices(v) for v in newVerts]
    return newVerts


  def globalCompEstimate(self) -> np.ndarray:
    return  getVertsFromBwComps(getBwComps(self.image, self.minCompSz))

  def getCroppedImg(self, verts: FRVertices, margin: int) -> (np.ndarray, np.ndarray):
    verts = verts.astype(int)
    img_np = self.image
    compCoords = np.vstack([verts.min(0), verts.max(0)])
    compCoords = getClippedBbox(img_np.shape, compCoords, margin).flatten()
    croppedImg = self.image[compCoords[1]:compCoords[3], compCoords[0]:compCoords[2], :]
    return croppedImg, compCoords