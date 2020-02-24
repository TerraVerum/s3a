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
  CLS_BASIC       : FRParam = newParam('Basic Shapes')

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
                        bgVerts: FRVertices = None) -> FRVertices:
    # TODO: Make this code more robust
    needsInvert = False
    if fgVerts is None:
      # If given background points, invert the component and grow the background
      fgVerts = bgVerts
      needsInvert = True
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
      centeredFgVerts = nanConcatList(getVertsFromBwComps(tmpBwShape,
                                                          simplifyVerts=False))

    newRegion = growSeedpoint(croppedImg, centeredFgVerts, self.seedThresh, self.minCompSz)
    newRegion = morphology.opening(newRegion, morphology.square(3))

    if needsInvert:
      newRegion = np.invert(newRegion)
    prevCompMask[cropOffset[1]:cropOffset[3], cropOffset[0]:cropOffset[2]] |= newRegion

    # Remember to account for the vertex offset
    newVerts = getVertsFromBwComps(prevCompMask)
    # TODO: pyqt crashses when sending signal with nan values. Find out why? in the
    #  meantime only use the largest component
    maxLenList = []
    for vertList in newVerts:
      if len(vertList) > len(maxLenList): maxLenList = vertList
    #for vertList in newVerts:
    #vertList += cropOffset[0:2]
    return FRVertices(maxLenList)


  def globalCompEstimate(self) -> np.ndarray:
    return  getVertsFromBwComps(getBwComps(self.image, self.minCompSz))

  def getCroppedImg(self, verts: FRVertices, margin: int) -> (np.ndarray, np.ndarray):
    verts = verts.nonNanEntries().astype(int)
    img_np = self.image
    compCoords = np.vstack([verts.min(0), verts.max(0)])
    compCoords = getClippedBbox(img_np.shape, compCoords, margin).flatten()
    croppedImg = self.image[compCoords[1]:compCoords[3], compCoords[0]:compCoords[2], :]
    return croppedImg, compCoords

@FR_SINGLETON.algParamMgr_.registerClass(IMPLS.CLS_REGION_GROW)
class BasicShapes(FRImageProcessor):
  def globalCompEstimate(self) -> np.ndarray:
    return np.zeros(self.image.shape, dtype=bool)

  def localCompEstimate(self, prevCompMask: np.ndarray, fgVerts: FRVertices=None, bgVerts: FRVertices=None) -> \
      np.ndarray:
    # Convert indices into boolean index masks
    fgMask = np.zeros(self.image.shape[0:2], dtype='uint8')
    cv.fillPoly(fgMask, )
