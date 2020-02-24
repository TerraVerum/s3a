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
from .processing import getVertsFromBwComps


@dataclass
class _FRDefaultAlgImpls(FRParamGroup):
  CLS_REGION_GROW : FRParam = newParam('Region Growing')

  PROP_MAIN_IMG_SEED_THRESH: FRParam = newParam('Seedpoint Threshold in Main Image', 10.)
  PROP_MIN_COMP_SZ: FRParam = newParam('Minimum New Component Size (px)', 50)
  PROP_NEW_COMP_SZ: FRParam = newParam('New Component Side Length (px)', 30)

IMPLS = _FRDefaultAlgImpls()

@FR_SINGLETON.algParamMgr_.registerClass(IMPLS.CLS_REGION_GROW)
class RegionGrow(FRImageProcessor):
  @FR_SINGLETON.algParamMgr_.registerProp(IMPLS.PROP_NEW_COMP_SZ)
  def newCompSz(self): pass
  @FR_SINGLETON.algParamMgr_.registerProp(IMPLS.PROP_MIN_COMP_SZ)
  def minCompSz(self): pass
  @FR_SINGLETON.algParamMgr_.registerProp(IMPLS.PROP_MAIN_IMG_SEED_THRESH)
  def mainImgSeedThresh(self): pass


  def localCompEstimate(self, prevCompMask: np.ndarray, fgVerts: FRVertices = None,
                        bgVerts: FRVertices = None) -> np.ndarray:
    # TODO: Make this code more robust
    needsInvert = False
    if fgVerts is None:
      # If given background points, invert the component and grow the background
      fgVerts = bgVerts
      needsInvert = True
    croppedImg, cropOffset = self.getCroppedImg(fgVerts, self.newCompSz)
    if croppedImg.size == 0:
      return prevCompMask
    offset = np.array([fgVerts.min(0)], dtype=int)
    centeredFgVerts = fgVerts - offset

    if fgVerts.connected:
      # Use all vertex points, not just the defined corners
      fillPolyArg = splitListAtNans(centeredFgVerts)
      tmpImg = np.zeros(croppedImg.shape[0:2], dtype='uint8')
      tmpBwShape = cv.fillPoly(tmpImg, fillPolyArg, 1)
      if np.prod(croppedImg.shape[0:2]) > 25e3:
        # Performance for using all bounds is prohibitive for large components
        approxMethod = cv.CHAIN_APPROX_SIMPLE
      else:
        approxMethod = cv.CHAIN_APPROX_NONE
      contours, _ = cv.findContours(tmpBwShape, cv.RETR_EXTERNAL, approxMethod)
      if len(contours) == 0:
        return prevCompMask
      centeredFgVerts = FRVertices(nanConcatList([contours[0][:,0,:]]))
    # TODO: Find a better method of determining whether to use all bounds
    newRegion = growSeedpoint(croppedImg, centeredFgVerts, self.mainImgSeedThresh, self.minCompSz)
    newRegion = morphology.opening(newRegion, morphology.square(3))

    if needsInvert:
      newRegion = np.invert(newRegion)

      # Remember to account for the vertex offset
    newVerts = getVertsFromBwComps(newRegion)
    for vertList in newVerts:
      vertList += offset + cropOffset
    return newVerts


  def globalCompEstimate(self) -> np.ndarray:
    return np.zeros(self.image.shape)

  def getCroppedImg(self, verts: FRVertices, margin: int) -> (np.ndarray, np.ndarray):
    verts = verts.astype(int)
    img_np = self.image
    compCoords = np.vstack([verts.min(0), verts.max(0)])
    compCoords = getClippedBbox(img_np.shape, compCoords, margin).flatten()
    croppedImg = self.image[compCoords[1]:compCoords[3], compCoords[0]:compCoords[2], :]
    return croppedImg, compCoords[0:2]