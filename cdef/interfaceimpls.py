from dataclasses import dataclass
from typing import List

import cv2 as cv
import numpy as np
from skimage.filters import gaussian
from skimage.measure import regionprops, label
from skimage.morphology import opening, closing, disk
from skimage.segmentation import active_contour

from cdef.processingutils import cornersToFullBoundary
from cdef.structures.typeoverloads import BlackWhiteImg
from .frgraphics.parameditors import FR_SINGLETON
from .generalutils import getClippedBbox
from .generalutils import splitListAtNans
from .interfaces import FRImageProcessor
from .processingutils import getVertsFromBwComps, getBwComps
from .processingutils import growSeedpoint, rmSmallComps
from .structures import FRParam, FRParamGroup, newParam
from .structures import FRVertices, FRComplexVertices


# For the purposes of processor impl's, a dataclass is probably not necessary. But I used it everywhere else
# so I'll stick to the pattern
@dataclass
class _FRDefaultAlgImpls(FRParamGroup):
  CLS_REGION_GROW : FRParam = newParam('Region Growing')
  CLS_SHAPES      : FRParam = newParam('Basic Shapes')
  CLS_SQUARES     : FRParam = newParam('Only Squares')
  CLS_BASIC       : FRParam = newParam('Shared Impl. Functionality')
  CLS_ACT_CONTOUR : FRParam = newParam('Active Contour -- SLOW')

  PROP_SEED_THRESH    : FRParam = newParam('Seedpoint Threshold', 10.)
  PROP_MIN_COMP_SZ    : FRParam = newParam('Minimum New Component Pixels', 50)
  PROP_MARGIN         : FRParam = newParam('Max Seedpoint Side Length', 30, None,
                                           'During seedpoint growth, the resulting '
                                           'component can grow very large if not restricted. '
                                           'Enforcing a larest size requirement limits '
                                           'these drawbacks.')
  PROP_ALLOW_MULT_REG : FRParam = newParam('Allow Noncontiguous Vertices', False)
  PROP_ALLOW_HOLES    : FRParam = newParam('Allow Holes in Component', False)
  PROP_STREL_SZ       : FRParam = newParam('Open->Close Struct. El. Width', 3)
  # Default values below retrieved from
  PROP_GAUS_SIGMA     : FRParam = newParam('Blurring Sigma', 3)
  PROP_ACT_CONT_ALPHA : FRParam = newParam('Alpha', 1)
  PROP_ACT_CONT_BETA  : FRParam = newParam('Beta', 0.1)
  PROP_ACT_CONT_GAMMA : FRParam = newParam('Gamma', 0.1)
  PROP_ACT_CONT_EDGE : FRParam = newParam('Edge Attraction', 5)
  PROP_N_A            : FRParam = newParam('No Editable Properties', None, 'none')


IMPLS = _FRDefaultAlgImpls()

@FR_SINGLETON.algParamMgr.registerClass(IMPLS.CLS_BASIC, addToList=False)
class FRBasicImageProcessorImpl(FRImageProcessor):
  @FR_SINGLETON.algParamMgr.registerProp(IMPLS.PROP_MIN_COMP_SZ)
  def minCompSz(self): pass
  @FR_SINGLETON.algParamMgr.registerProp(IMPLS.PROP_MARGIN)
  def margin(self): pass
  @FR_SINGLETON.algParamMgr.registerProp(IMPLS.PROP_ALLOW_MULT_REG)
  def allowMultReg(self): pass
  @FR_SINGLETON.algParamMgr.registerProp(IMPLS.PROP_ALLOW_HOLES)
  def allowHoles(self): pass
  @FR_SINGLETON.algParamMgr.registerProp(IMPLS.PROP_STREL_SZ)
  def strelSz(self): pass
  

  def localCompEstimate(self, prevCompMask: BlackWhiteImg, fgVerts: FRVertices = None, bgVerts: FRVertices = None) -> \
      BlackWhiteImg:
    """
    Performs basic operations shared by all FICS-specified image processors. That is, whether they allow holes,
    regions comprised of multiple separate segments, their minimum size, and margin around specified vertices.
    """
    if np.count_nonzero(prevCompMask) <= 1: return rmSmallComps(prevCompMask, self.minCompSz)
    if not self.allowHoles:
      # Fill in outer contours
      tmpVerts = getVertsFromBwComps(prevCompMask, externOnly=True)
      tmpMask = prevCompMask.astype('uint8')
      prevCompMask = cv.fillPoly(tmpMask, tmpVerts, 1) > 0
    if not self.allowMultReg:
      # Take out all except the largest region
      regions = regionprops(label(prevCompMask))
      if len(regions) == 0: return prevCompMask
      maxRegion = regions[0]
      for region in regions:
        if region.area > maxRegion.area:
          maxRegion = region
      prevCompMask[:] = False
      prevCompMask[maxRegion.coords[:,0], maxRegion.coords[:,1]] = True
    # Remember to account for the vertex offset
    return rmSmallComps(prevCompMask, self.minCompSz)

  def globalCompEstimate(self) -> List[FRComplexVertices]:
    initialList = getVertsFromBwComps(getBwComps(self.image, self.minCompSz), externOnly=True)
    return [FRComplexVertices([lst]) for lst in initialList]


@FR_SINGLETON.algParamMgr.registerClass(IMPLS.CLS_REGION_GROW)
class FRRegionGrow(FRBasicImageProcessorImpl):
  @FR_SINGLETON.algParamMgr.registerProp(IMPLS.PROP_SEED_THRESH)
  def seedThresh(self): pass

  def localCompEstimate(self, prevCompMask: BlackWhiteImg, fgVerts: FRVertices = None,
                        bgVerts: FRVertices = None) -> BlackWhiteImg:
    if prevCompMask is None:
      prevCompMask = np.zeros(self.image.shape[:2], dtype=bool)
    else:
      # Don't modify the original version
      prevCompMask = prevCompMask.copy()
    # TODO: Make this code more robust
    # -----
    # DETERMINE BITWISE RELATIONSHIP B/W OLD AND NEW MASKS
    # -----
    if fgVerts is None:
      # Add to background
      bitOperation = lambda curRegion, other: curRegion & (~other)
      fgVerts = bgVerts
    else:
      # Add to foreground
      bitOperation = np.bitwise_or

    # -----
    # DETERMINE INWARD/OUTWARD GROWTH BASED ON VERTEX SHAPE
    # -----
    if np.all(fgVerts == fgVerts[0,:]):
      # Remove unnecessary redundant seedpoints
      fgVerts = fgVerts[[0],:]
    if fgVerts.shape[0] == 1:
      # Grow outward
      growFunc = growSeedpoint
      compMargin = self.margin
      fgVerts.connected = False
    else:
      # Grow inward
      growFunc = lambda *args: ~growSeedpoint(*args)
      compMargin = 0
    croppedImg, cropOffset = self.getCroppedImg(fgVerts, compMargin)
    if croppedImg.size == 0:
      return prevCompMask
    centeredFgVerts = fgVerts - cropOffset[0:2]

    filledMask = None
    tmpImgToFill = np.zeros(croppedImg.shape[0:2], dtype='uint8')

    # For small enough shapes, get all boundary pixels instead of just shape vertices
    if centeredFgVerts.connected:
      centeredFgVerts = cornersToFullBoundary(centeredFgVerts, 50e3)

    newRegion = growFunc(croppedImg, centeredFgVerts, self.seedThresh)
    if fgVerts.connected:
      # For connected vertices, zero out region locations outside the user defined area
      if filledMask is None:
        filledMask = cv.fillPoly(tmpImgToFill, [centeredFgVerts], 1) > 0
      newRegion[~filledMask] = False

    rowColSlices = (slice(cropOffset[1], cropOffset[3]),
                    slice(cropOffset[0], cropOffset[2]))
    prevCompMask[rowColSlices] = bitOperation(prevCompMask[rowColSlices], newRegion)
    openCloseStrel = disk(self.strelSz)
    prevCompMask = opening(closing(prevCompMask, openCloseStrel), openCloseStrel)
    return super().localCompEstimate(prevCompMask)

  def getCroppedImg(self, verts: FRVertices, margin: int) -> (np.ndarray, np.ndarray):
    verts = np.vstack(verts)
    img_np = self.image
    compCoords = np.vstack([verts.min(0), verts.max(0)])
    compCoords = getClippedBbox(img_np.shape, compCoords, margin).flatten()
    croppedImg = self.image[compCoords[1]:compCoords[3], compCoords[0]:compCoords[2], :]
    return croppedImg, compCoords

@FR_SINGLETON.algParamMgr.registerClass(IMPLS.CLS_SHAPES)
class FRBasicShapes(FRBasicImageProcessorImpl):
  def localCompEstimate(self, prevCompMask: BlackWhiteImg, fgVerts: FRVertices=None, bgVerts: FRVertices=None) -> \
      BlackWhiteImg:
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
    return super().localCompEstimate(prevCompMask)


@FR_SINGLETON.algParamMgr.registerClass(IMPLS.CLS_SQUARES)
class FROnlySquares(FRBasicShapes):
  def globalCompEstimate(self) -> List[FRComplexVertices]:
    polyVerts = super().globalCompEstimate()
    outVerts = []
    for vertList in polyVerts:
      squareCorners = np.vstack([vertList.stack().min(0), vertList.stack().max(0)])
      # Turn square diagonals into proper vertices
      colIdx = [0,1,0,1,0,1,0,1]
      rowIdx = [0,0,0,1,1,1,1,0]
      squareVerts = squareCorners[rowIdx, colIdx].reshape(-1,2)
      outVerts.append(FRComplexVertices([squareVerts]))
    return outVerts

  def localCompEstimate(self, prevCompMask: np.ndarray, fgVerts: FRVertices = None, bgVerts: FRVertices = None) -> \
      np.ndarray:
    # Convert indices into boolean index masks
    compMask = super().localCompEstimate(prevCompMask, fgVerts, bgVerts)
    outMask = np.zeros(compMask.shape, dtype=bool)
    for region in regionprops(label(compMask)):
      outMask[region.bbox[0]:region.bbox[2],region.bbox[1]:region.bbox[3]] = True
    return outMask    

@FR_SINGLETON.algParamMgr.registerClass(IMPLS.CLS_ACT_CONTOUR)
class FRActiveContour(FRBasicImageProcessorImpl):
  @FR_SINGLETON.algParamMgr.registerProp(IMPLS.PROP_GAUS_SIGMA)
  def blurSigma(self): pass
  @FR_SINGLETON.algParamMgr.registerProp(IMPLS.PROP_ACT_CONT_ALPHA)
  def alpha(self): pass
  @FR_SINGLETON.algParamMgr.registerProp(IMPLS.PROP_ACT_CONT_BETA)
  def beta(self): pass
  @FR_SINGLETON.algParamMgr.registerProp(IMPLS.PROP_ACT_CONT_GAMMA)
  def gamma(self): pass
  @FR_SINGLETON.algParamMgr.registerProp(IMPLS.PROP_ACT_CONT_EDGE)
  def wEdge(self): pass

  def localCompEstimate(self, prevCompMask: BlackWhiteImg, fgVerts: FRVertices = None, bgVerts: FRVertices = None) -> \
      BlackWhiteImg:
    blurredImg = gaussian(self.image, self.blurSigma)
    prevCompMask = prevCompMask.copy()
    if fgVerts is None:
      # Snake to get area, then subtract
      contourVertsArg = bgVerts
      isBg = True
    else:
      isBg = False
      contourVertsArg = fgVerts

    # Get high precision by retrieving all boundary pixels
    contourVertsArg = cornersToFullBoundary(contourVertsArg)
    # 'xy' option will soon be removed, so convert to rc
    # verts are mutated into bad shape during active_contouring processing, so cast to regular array
    contourVertsArg = contourVertsArg.asRowCol().view(np.ndarray)
    contouredVerts = active_contour(blurredImg, contourVertsArg, self.alpha, self.beta,
                                    w_edge=self.wEdge, gamma=self.gamma, coordinates='rc')
    out = np.zeros(self.image.shape[:2], dtype='uint8')
    # Format contour verts for fillPoly
    contouredVerts = np.fliplr(contouredVerts.astype('int'))
    contourRegion = cv.fillPoly(out, [contouredVerts], 1).astype(bool)
    if isBg:
      prevCompMask = prevCompMask & ~contourRegion
    else:
      prevCompMask |= contourRegion
    return super().localCompEstimate(prevCompMask, fgVerts, bgVerts)
