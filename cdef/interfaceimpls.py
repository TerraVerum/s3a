from typing import Tuple, Optional

import cv2 as cv
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.measure import regionprops, label

from cdef.generalutils import splitListAtNans
from cdef.processingutils import growSeedpoint, cornersToFullBoundary, getCroppedImg, \
  area_coord_regionTbl
from cdef.structures import BlackWhiteImg, FRVertices
from imageprocessing import algorithms
from imageprocessing.algorithms import watershedProcess
from imageprocessing.common import Image
from imageprocessing.processing import ImageIO, ImageProcess


def cropImgToROI(margin=10):
  io = ImageIO.createFrom(cropImgToROI, locals())
  proc = io.initProcess('Crop to Vertices')
  def crop_to_verts(_image: Image, _fgVerts: FRVertices, _bgVerts: FRVertices,
                    _prevCompMask: BlackWhiteImg, _margin=margin):
    maskLocs = np.nonzero(_prevCompMask)
    maskCoords = np.hstack([m[:,None] for m in maskLocs])
    if maskCoords.size > 0:
      maskBbox = np.vstack([maskCoords.min(0), maskCoords.max(0)])
    else:
      maskBbox = None

    asForeground = True
    allVerts = []
    if _fgVerts is None and _bgVerts is None:
      # Give whole image as input
      shape = _image.shape[:2][::-1]
      _fgVerts = FRVertices([[0,0], [0, shape[1]],
                             [shape[0], shape[1]], [shape[0], 0]
                            ])
    if _fgVerts is None:
      asForeground = False
      _fgVerts = _bgVerts
      allVerts.append(_fgVerts)
      _bgVerts = None
    else:
      allVerts.append(_fgVerts)
    if _bgVerts is not None:
      allVerts.append(_bgVerts)

    allVerts = np.vstack(allVerts)
    cropped, bounds = getCroppedImg(_image, allVerts, _margin, maskBbox)
    vertOffset = bounds.min(0)
    for vertList in _fgVerts, _bgVerts:
      if vertList is not None:
        vertList -= vertOffset
    boundSlices = slice(*bounds[:,1]), slice(*bounds[:,0])
    _prevCompMask = _prevCompMask[boundSlices]
    return ImageIO(image=cropped, fgVerts=_fgVerts, bgVerts=_bgVerts,
                   boundSlices=boundSlices, origImg=_image, asForeground=asForeground)
  proc.addFunction(crop_to_verts)
  return proc

def updateCroppedArea():
  proc = ImageProcess('Update Cropped Area', ImageIO())
  def update_area(_image: Image, _origImg: Image, _boundSlices: Tuple[slice], _asForeground: bool,
                  _prevCompMask):
    _prevCompMask = _prevCompMask.copy()
    if _asForeground:
      bitOperation = np.bitwise_or
    else:
      # Add to background
      bitOperation = lambda curRegion, other: curRegion & (~other)

    _prevCompMask[_boundSlices] = bitOperation(_prevCompMask[_boundSlices], _image)
    return ImageIO(image=_prevCompMask)
  proc.addFunction(update_area)
  return proc


def fillHoles():
  io = ImageIO.createFrom(fillHoles, locals())
  proc = io.initProcess('Fill Holes')

  def fill_holes(_image: Image):
    return ImageIO(image=binary_fill_holes(_image))
  proc.addFunction(fill_holes)
  return proc

def openClose():
  io = ImageIO.createFrom(openClose, locals())
  proc = io.initProcess('Open -> Close')
  def cvt_to_uint(_image: Image):
    return ImageIO(image = _image.astype('uint8'))
  proc.addFunction(cvt_to_uint)
  p = algorithms.morphologyExProcess(cv.MORPH_OPEN)
  proc.addProcess(p)
  proc.addProcess(algorithms.morphologyExProcess(cv.MORPH_CLOSE))
  return proc

def keepLargestConnComp():
  io = ImageIO.createFrom(keepLargestConnComp, locals())
  proc = io.initProcess('Keep Largest Connected Components')

  def keep_largest_comp(_image: Image):
    _regionPropTbl = area_coord_regionTbl(_image)
    out = np.zeros(_image.shape, bool)
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
  def rm_small_comp(_image: Image, _minSzThreshold=minSzThreshold):
    _regionPropTbl = area_coord_regionTbl(_image)
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
  proc.addProcess(openClose())
  proc.addProcess(keepLargestConnComp())
  proc.addProcess(removeSmallConnComps())
  return proc

class FRTopLevelProcessors:
  @staticmethod
  def regionGrowProcessor(margin=5, seedThresh=10):
    curLocals = locals()
    io = ImageIO.createFrom(FRTopLevelProcessors.regionGrowProcessor, locals())
    proc = io.initProcess('Region Growing')
    def region_grow(_image: Image, _prevCompMask: BlackWhiteImg, _fgVerts: FRVertices,
                    _bgVerts: FRVertices, _seedThresh=seedThresh, _margin=margin):
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
      croppedImg, cropOffset = getCroppedImg(_image, _fgVerts, compMargin)
      if croppedImg.size == 0:
        return ImageIO(image=_prevCompMask)
      centeredFgVerts = _fgVerts - cropOffset[0,:]

      tmpImgToFill = np.zeros(croppedImg.shape[0:2], dtype='uint8')

      # For small enough shapes, get all boundary pixels instead of just shape vertices
      if centeredFgVerts.connected:
        centeredFgVerts = cornersToFullBoundary(centeredFgVerts, 50e3)

      newRegion = growFunc(croppedImg, centeredFgVerts, _seedThresh)
      if _fgVerts.connected:
        # For connected vertices, zero out region locations outside the user defined area
        filledMask = cv.fillPoly(tmpImgToFill, [centeredFgVerts], 1) > 0
        newRegion[~filledMask] = False

      rowColSlices = (slice(cropOffset[0,1], cropOffset[1,1]),
                      slice(cropOffset[0,0], cropOffset[1,0]))
      _prevCompMask[rowColSlices] = bitOperation(_prevCompMask[rowColSlices], newRegion)
      return ImageIO(image=_prevCompMask)
    proc.addFunction(region_grow)
    proc.addProcess(basicOpsCombo())
    return proc

  @staticmethod
  def regionGrowProcessor2(seedThresh=10):
    curLocals = locals()
    io = ImageIO.createFrom(FRTopLevelProcessors.regionGrowProcessor2, locals())
    proc = io.initProcess('Region Growing 2')
    proc.addProcess(cropImgToROI())

    def region_grow(_image: Image, _prevCompMask: BlackWhiteImg, _fgVerts: FRVertices,
                    _asForeground: bool, _seedThresh=seedThresh):
      if _image.size == 0:
        return ImageIO(image=_prevCompMask)
      if _prevCompMask is None:
        _prevCompMask = np.zeros(_image.shape[:2], dtype=bool)
      # -----
      # DETERMINE BITWISE RELATIONSHIP B/W OLD AND NEW MASKS
      # -----
      if np.all(_fgVerts == _fgVerts[0,:]):
        # Remove unnecessary redundant seedpoints
        _fgVerts = _fgVerts[[0],:]
      if _fgVerts.shape[0] == 1:
        # Grow outward
        growFunc = growSeedpoint
        _fgVerts.connected = False
      else:
        # Grow inward
        growFunc = lambda *args: ~growSeedpoint(*args)

      tmpImgToFill = np.zeros(_image.shape[0:2], dtype='uint8')
      # For small enough shapes, get all boundary pixels instead of just shape vertices
      if _fgVerts.connected:
        _fgVerts = cornersToFullBoundary(_fgVerts, 50e3)

      newRegion = growFunc(_image, _fgVerts, _seedThresh)
      if _fgVerts.connected:
        # For connected vertices, zero out region locations outside the user defined area
        filledMask = cv.fillPoly(tmpImgToFill, [_fgVerts], 1) > 0
        newRegion[~filledMask] = False

      return ImageIO(image=newRegion)
    proc.addFunction(region_grow)
    proc.addProcess(updateCroppedArea())
    proc.addProcess(basicOpsCombo())
    return proc

  @staticmethod
  def basicShapesProcessor():
    io = ImageIO.createFrom(FRTopLevelProcessors.basicShapesProcessor, locals())
    proc = io.initProcess('Basic Shapes')

    def get_basic_shapes(_image: Image, _prevCompMask: BlackWhiteImg,
                         _fgVerts: FRVertices, _bgVerts: FRVertices):
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

  @staticmethod
  def onlySquaresProcessor():
    io = ImageIO.createFrom(FRTopLevelProcessors.basicShapesProcessor, locals())
    proc = io.initProcess('Only Squares')

    proc.addProcess(FRTopLevelProcessors.basicShapesProcessor())
    def convert_to_squares(_image: Image):
      outMask = np.zeros(_image.shape, dtype=bool)
      for region in regionprops(label(_image)):
        outMask[region.bbox[0]:region.bbox[2],region.bbox[1]:region.bbox[3]] = True
      return ImageIO(image=outMask)
    proc.addFunction(convert_to_squares)
    return proc

  # @staticmethod
  # def watershedProcessor():
  #   return watershedProcess()