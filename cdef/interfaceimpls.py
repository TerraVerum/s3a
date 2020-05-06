from functools import wraps

import cv2 as cv
import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes
from skimage import morphology
from skimage.measure import regionprops, label
from scipy.ndimage import binary_opening, binary_closing

from cdef.generalutils import splitListAtNans
from cdef.processingutils import growSeedpoint, cornersToFullBoundary, _getCroppedImg, \
  _area_coord_regionTbl
from cdef.structures import BlackWhiteImg, FRVertices
from imageprocessing import algorithms
from imageprocessing.common import Image
from imageprocessing.processing import ImageIO

def fillHoles():
  io = ImageIO.createFrom(fillHoles, locals())
  proc = io.initProcess('Fill Holes')

  def fill_holes(_image: Image):
    return ImageIO(image=binary_fill_holes(_image.data))
  proc.addFunction(fill_holes)
  return proc

def closeOpen():
  io = ImageIO.createFrom(closeOpen, locals())
  proc = io.initProcess('Close -> Open Compondent')
  def cvt_to_uint(_image: Image):
    return ImageIO(image = _image.data.astype('uint8'))
  proc.addFunction(cvt_to_uint)
  proc.addProcess(algorithms.morphologyExProcess(cv.MORPH_CLOSE))
  proc.addProcess(algorithms.morphologyExProcess(cv.MORPH_OPEN))
  return proc

def keepLargestConnComp():
  io = ImageIO.createFrom(keepLargestConnComp, locals())
  proc = io.initProcess('Keep Largest Connected Components')

  def keep_largest_comp(_image: Image, _regionPropTbl: pd.DataFrame):
      if _regionPropTbl is None:
        _regionPropTbl = _area_coord_regionTbl(_image)
      out = np.zeros(_image.data.shape, bool)
      coords = _regionPropTbl.coords[_regionPropTbl.area.argmax()]
      if coords.size == 0:
        return ImageIO(image=out)
      out[coords[:,0], coords[:,1]] = True
      return ImageIO(image=out)
  proc.addFunction(keep_largest_comp)
  return proc

def removeSmallConnComps(minSzThreshold=30):
  io = ImageIO.createFrom(removeSmallConnComps, locals())
  proc = io.initProcess('Remove Conn. Comp. Smaller Than...')
  def rm_small_comp(_image: Image, _regionPropTbl: pd.DataFrame, _minSzThreshold):
    if _regionPropTbl is None:
      _regionPropTbl = _area_coord_regionTbl(_image)
    validCoords = _regionPropTbl.coords[_regionPropTbl.area >= _minSzThreshold]
    out = np.zeros(_image.shape, bool)
    if len(validCoords) == 0:
      return ImageIO(image=out)
    coords = np.vstack(validCoords)
    out[coords[:,0], coords[:,1]] = True
    return ImageIO(image=out)
  proc.addFunction(rm_small_comp)
  return proc

def basicOpsCombo():
  io = ImageIO.createFrom(basicOpsCombo, locals())
  proc = io.initProcess('Basic Region Operations')
  proc.addProcess(fillHoles())
  proc.addProcess(closeOpen())
  proc.addProcess(keepLargestConnComp())
  proc.addProcess(removeSmallConnComps())
  return proc

def regionGrowProcessor(margin=5, seedThresh=10, strelSz=3):
  curLocals = locals()
  io = ImageIO.createFrom(regionGrowProcessor, locals())
  proc = io.initProcess('Region Growing')
  def region_grow(_image: Image, _prevCompMask: BlackWhiteImg, _fgVerts: FRVertices,
                  _bgVerts: FRVertices, _seedThresh=seedThresh, _margin=margin, _strelSz=strelSz):
    if _prevCompMask is None:
      _prevCompMask = np.zeros(_image.shape[:2], dtype=bool)
    else:
      # Don't modify the original version
      _prevCompMask = _prevCompMask.copy()
    # TODO: Make this code more robust
    # -----
    # DETERMINE BITWISE RELATIONSHIP B/W OLD AND NEW MASKS
    # -----
    if _fgVerts is None:
      # Add to background
      bitOperation = lambda curRegion, other: curRegion & (~other)
      _fgVerts = _bgVerts
    else:
      # Add to foreground
      bitOperation = np.bitwise_or

    # -----
    # DETERMINE INWARD/OUTWARD GROWTH BASED ON VERTEX SHAPE
    # -----
    if np.all(_fgVerts == _fgVerts[0,:]):
      # Remove unnecessary redundant seedpoints
      _fgVerts = _fgVerts[[0],:]
    if _fgVerts.shape[0] == 1:
      # Grow outward
      growFunc = growSeedpoint
      compMargin = _margin
      _fgVerts.connected = False
    else:
      # Grow inward
      growFunc = lambda *args: ~growSeedpoint(*args)
      compMargin = 0
    croppedImg, cropOffset = _getCroppedImg(_image.data, _fgVerts, compMargin)
    if croppedImg.size == 0:
      return ImageIO(image=_prevCompMask)
    centeredFgVerts = _fgVerts - cropOffset[0:2]

    filledMask = None
    tmpImgToFill = np.zeros(croppedImg.shape[0:2], dtype='uint8')

    # For small enough shapes, get all boundary pixels instead of just shape vertices
    if centeredFgVerts.connected:
      centeredFgVerts = cornersToFullBoundary(centeredFgVerts, 50e3)

    newRegion = growFunc(croppedImg, centeredFgVerts, _seedThresh)
    if _fgVerts.connected:
      # For connected vertices, zero out region locations outside the user defined area
      if filledMask is None:
        filledMask = cv.fillPoly(tmpImgToFill, [centeredFgVerts], 1) > 0
      newRegion[~filledMask] = False

    rowColSlices = (slice(cropOffset[1], cropOffset[3]),
                    slice(cropOffset[0], cropOffset[2]))
    _prevCompMask[rowColSlices] = bitOperation(_prevCompMask[rowColSlices], newRegion)
    openCloseStrel = morphology.disk(_strelSz)
    _prevCompMask = morphology.opening(morphology.closing(_prevCompMask, openCloseStrel), openCloseStrel)
    return ImageIO(image=_prevCompMask)
  proc.addFunction(region_grow)
  proc.addProcess(basicOpsCombo())
  return proc

def basicShapesProcessor():
  io = ImageIO.createFrom(basicShapesProcessor, locals())
  proc = io.initProcess('Basic Shapes')

  def get_basic_shapes(_image: Image, _prevCompMask: BlackWhiteImg,
                       _fgVerts: FRVertices=None, _bgVerts: FRVertices=None):
    # Don't modify the original version
    prevCompMask = _prevCompMask.copy()
    # Convert indices into boolean index masks
    masks = []
    for ii, verts in enumerate((_fgVerts, _bgVerts)):
      curMask = np.zeros(_image.shape[0:2], dtype='uint8')
      if verts is not None:
        fillPolyArg = splitListAtNans(verts)
        curMask = cv.fillPoly(curMask, fillPolyArg, 1)
      masks.append(curMask.astype(bool))
    # Foreground is additive, bg is subtractive. If both fg and bg are present, default to keeping old value
    addRegion = masks[0] & ~masks[1]
    subRegion = ~masks[0] & masks[1]
    prevCompMask |= addRegion
    prevCompMask &= (~subRegion)
    return ImageIO(image=prevCompMask)
  proc.addFunction(get_basic_shapes)
  proc.addProcess(basicOpsCombo())
  return proc

def onlySquaresProcessor():
  io = ImageIO.createFrom(basicShapesProcessor, locals())
  proc = io.initProcess('Only Squares')

  proc.addProcess(basicShapesProcessor())
  def convert_to_squares(_image: Image):
    outMask = np.zeros(_image.shape, dtype=bool)
    for region in regionprops(label(_image.data)):
      outMask[region.bbox[0]:region.bbox[2],region.bbox[1]:region.bbox[3]] = True
    return ImageIO(image=outMask)
  proc.addFunction(convert_to_squares)
  return proc