from dataclasses import dataclass

import cv2 as cv
import numpy as np
from skimage import morphology
from typing import List

from Annotator.generalutils import splitListAtNans, nanConcatList, largestList
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
  CLS_SQUARES     : FRParam = newParam('Only Squares')

  PROP_SEED_THRESH: FRParam = newParam('Seedpoint Threshold in Main Image', 10.)
  PROP_MIN_COMP_SZ: FRParam = newParam('Minimum New Component Size (px)', 50)
  PROP_NEW_COMP_SZ: FRParam = newParam('New Component Side Length (px)', 30)
  PROP_MARGIN     : FRParam = newParam('New Component Margin (px)', 5)
  PROP_N_A     : FRParam = newParam('No Editable Properties', None, 'none')

IMPLS = _FRDefaultAlgImpls()

@FR_SINGLETON.algParamMgr.registerClass(IMPLS.CLS_REGION_GROW)
class RegionGrow(FRImageProcessor):
  @FR_SINGLETON.algParamMgr.registerProp(IMPLS.PROP_NEW_COMP_SZ)
  def newCompSz(self): pass
  @FR_SINGLETON.algParamMgr.registerProp(IMPLS.PROP_MIN_COMP_SZ)
  def minCompSz(self): pass
  @FR_SINGLETON.algParamMgr.registerProp(IMPLS.PROP_MARGIN)
  def margin(self): pass
  @FR_SINGLETON.algParamMgr.registerProp(IMPLS.PROP_SEED_THRESH)
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
      return largestList(getVertsFromBwComps(prevCompMask))
    centeredFgVerts = fgVerts - cropOffset[0:2]

    # For small enough shapes, get all boundary pixels instead of just shape vertices
    if fgVerts.connected and np.prod(croppedImg.shape[0:2]) < 50e3:
      # Use all vertex points, not just the defined corners
      fillPolyArg = splitListAtNans(centeredFgVerts)
      tmpImg = np.zeros(croppedImg.shape[0:2], dtype='uint8')
      tmpBwShape = cv.fillPoly(tmpImg, fillPolyArg, 1)
      centeredFgVerts = nanConcatList(getVertsFromBwComps(tmpBwShape,
                                                          simplifyVerts=False))

    newRegion = ~growSeedpoint(croppedImg, centeredFgVerts, self.seedThresh, self.minCompSz)
    newRegion = morphology.opening(newRegion, morphology.square(3))
    if needsInvert:
      prevCompMask[cropOffset[1]:cropOffset[3], cropOffset[0]:cropOffset[2]] |= (~newRegion)
    else:
      prevCompMask[cropOffset[1]:cropOffset[3], cropOffset[0]:cropOffset[2]] |= newRegion

    # Remember to account for the vertex offset
    newVerts = getVertsFromBwComps(prevCompMask)
    # TODO: pyqt crashses when sending signal with nan values. Find out why? in the
    #  meantime only use the largest component
    return largestList(newVerts)


  def globalCompEstimate(self) -> List[FRVertices]:
    return  getVertsFromBwComps(getBwComps(self.image, self.minCompSz))

  def getCroppedImg(self, verts: FRVertices, margin: int) -> (np.ndarray, np.ndarray):
    verts = verts.nonNanEntries().astype(int)
    img_np = self.image
    compCoords = np.vstack([verts.min(0), verts.max(0)])
    compCoords = getClippedBbox(img_np.shape, compCoords, margin).flatten()
    croppedImg = self.image[compCoords[1]:compCoords[3], compCoords[0]:compCoords[2], :]
    return croppedImg, compCoords

@FR_SINGLETON.algParamMgr.registerClass(IMPLS.CLS_BASIC)
class BasicShapes(FRImageProcessor):

  # If a class has no editable properties, this must be indicated for proper registration parsing
  @FR_SINGLETON.algParamMgr.registerProp(IMPLS.PROP_MIN_COMP_SZ)
  def minCompSz(self): pass

  def globalCompEstimate(self) -> List[FRVertices]:
    return getVertsFromBwComps(getBwComps(self.image, self.minCompSz))

  def localCompEstimate(self, prevCompMask: np.ndarray, fgVerts: FRVertices=None, bgVerts: FRVertices=None) -> \
      FRVertices:
    # Convert indices into boolean index masks
    masks = []
    for ii, verts in enumerate((fgVerts, bgVerts)):
      curMask = np.zeros(self.image.shape[0:2], dtype='uint8')
      if verts is not None:
        fillPolyArg = splitListAtNans(verts)
        curMask = cv.fillPoly(curMask, fillPolyArg, 1)
      masks.append(curMask.astype(bool))
    # Foreground is additive, bg is subtractive. If both fg and bg are present, default to keeping old value
    addRegion = masks[0] & ~masks[1]
    subRegion = ~masks[0] & masks[1]
    prevCompMask |= addRegion
    prevCompMask &= (~subRegion)
    newVerts = getVertsFromBwComps(prevCompMask)
    return largestList(newVerts)


@FR_SINGLETON.algParamMgr.registerClass(IMPLS.CLS_SQUARES)
class OnlySquares(BasicShapes):
  # Required for registering a processor that has no editable properties
  @FR_SINGLETON.algParamMgr.registerProp(IMPLS.PROP_N_A)
  def _(self): pass

  def globalCompEstimate(self) -> List[FRVertices]:
    polyVerts = getVertsFromBwComps(getBwComps(self.image, self.minCompSz))
    outVerts = []
    for vertList in polyVerts:
      squareVerts = np.vstack([vertList.min(0), vertList.max(0)])
      outVerts.append(FRVertices(squareVerts))
    return outVerts

  def localCompEstimate(self, prevCompMask: np.ndarray, fgVerts: FRVertices = None, bgVerts: FRVertices = None) -> \
      np.ndarray:
    # Convert indices into boolean index masks
    verts = super().localCompEstimate(prevCompMask, fgVerts, bgVerts).nonNanEntries()
    squareCoord = np.vstack([verts.min(0), verts.max(0)])

    return FRVertices([
      squareCoord[0,:],
      [squareCoord[0,0], squareCoord[1,1]],
      squareCoord[1,:],
      [squareCoord[1,0], squareCoord[0,1]],
    ])