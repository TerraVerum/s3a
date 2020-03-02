from dataclasses import dataclass

import cv2 as cv
import numpy as np
from skimage.measure import regionprops, label
from typing import List

from Annotator.generalutils import splitListAtNans, nanConcatList, largestList
from Annotator.params import FRParamGroup, FRParam, newParam, FRComplexVertices
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
  PROP_GROW_OUT   : FRParam = newParam('Grow outward', False)
  PROP_N_A        : FRParam = newParam('No Editable Properties', None, 'none')

  SHC_GROW_OUT    : FRParam = newParam('Grow Outward', 'Ctrl+G,O', 'none')
  SHC_GROW_IN    : FRParam = newParam('Grow Inward', 'Ctrl+G,I', 'none')

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
                        bgVerts: FRVertices = None) -> np.ndarray:
    # Don't modify the original version
    prevCompMask = prevCompMask.copy()
    # TODO: Make this code more robust
    if fgVerts is None:
      # Add to background
      bitOperation = lambda curRegion, other: curRegion & (~other)
      fgVerts = bgVerts
    else:
      # Add to foreground
      bitOperation = np.bitwise_or
    if len(fgVerts) == 1:
      # Grow outward
      growFunc = growSeedpoint
    else:
      # Grow inward
      growFunc = lambda *args: ~growSeedpoint(*args)
    croppedImg, cropOffset = self.getCroppedImg(fgVerts, self.margin)
    if croppedImg.size == 0:
      return prevCompMask
    centeredFgVerts = fgVerts - cropOffset[0:2]

    # For small enough shapes, get all boundary pixels instead of just shape vertices
    if fgVerts.connected and np.prod(croppedImg.shape[0:2]) < 50e3:
      # Use all vertex points, not just the defined corners
      tmpImg = np.zeros(croppedImg.shape[0:2], dtype='uint8')
      tmpBwShape = cv.fillPoly(tmpImg, [centeredFgVerts], 1)
      centeredFgVerts = getVertsFromBwComps(tmpBwShape,simplifyVerts=False).filledVerts()
      centeredFgVerts = np.vstack(centeredFgVerts)

    newRegion = growFunc(croppedImg, centeredFgVerts, self.seedThresh, self.minCompSz)
    rowColSlices = (slice(cropOffset[1], cropOffset[3]),
                    slice(cropOffset[0], cropOffset[2]))
    prevCompMask[rowColSlices] = bitOperation(prevCompMask[rowColSlices], newRegion)

    # Remember to account for the vertex offset
    return prevCompMask


  def globalCompEstimate(self) -> List[FRComplexVertices]:
    initialList = getVertsFromBwComps(getBwComps(self.image, self.minCompSz), externOnly=True)
    return [FRComplexVertices(lst) for lst in initialList]

  def getCroppedImg(self, verts: FRVertices, margin: int) -> (np.ndarray, np.ndarray):
    verts = np.vstack(verts)
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

  def globalCompEstimate(self) -> List[FRComplexVertices]:
    initialList = getVertsFromBwComps(getBwComps(self.image, self.minCompSz), externOnly=True)
    return [FRComplexVertices(lst) for lst in initialList]

  def localCompEstimate(self, prevCompMask: np.ndarray, fgVerts: FRVertices=None, bgVerts: FRVertices=None) -> \
      FRVertices:
    # Don't modify the original version
    prevCompMask = prevCompMask.copy()
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
    return prevCompMask


@FR_SINGLETON.algParamMgr.registerClass(IMPLS.CLS_SQUARES)
class OnlySquares(BasicShapes):
  def globalCompEstimate(self) -> List[FRComplexVertices]:
    polyVerts = getVertsFromBwComps(getBwComps(self.image, self.minCompSz), externOnly=True)
    outVerts = []
    for vertList in polyVerts:
      squareVerts = np.vstack([vertList.min(0), vertList.max(0)]).view(FRVertices)
      outVerts.append(FRComplexVertices(squareVerts))
    return outVerts

  def localCompEstimate(self, prevCompMask: np.ndarray, fgVerts: FRVertices = None, bgVerts: FRVertices = None) -> \
      np.ndarray:
    # Convert indices into boolean index masks
    compMask = super().localCompEstimate(prevCompMask, fgVerts, bgVerts)
    outMask = np.zeros(compMask.shape, dtype=bool)
    for region in regionprops(label(compMask)):
      outMask[region.bbox[0]:region.bbox[2],region.bbox[1]:region.bbox[3]] = True
    return outMask