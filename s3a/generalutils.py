import re
from ast import literal_eval
from contextlib import contextmanager
from inspect import isclass
from pathlib import Path
from typing import Any, Optional, List, Collection, Callable, Tuple, Union, Sequence
import typing as t

import numpy as np
import pandas as pd
from pandas import DataFrame as df
import cv2 as cv
from pyqtgraph.parametertree import Parameter

from s3a.constants import ANN_AUTH_DIR
from s3a.graphicsutils import yaml
from s3a.structures.typeoverloads import TwoDArr, FilePath
from .structures import XYVertices, FRParam, ComplexXYVertices, NChanImg


def stackedVertsPlusConnections(vertList: ComplexXYVertices) -> (XYVertices, np.ndarray):
  """
  Utility for concatenating all vertices within a list while recording where separations
  occurred
  """
  allVerts = [np.zeros((0,2))]
  separationIdxs = []
  idxOffset = 0
  for curVerts in vertList:
    allVerts.append(curVerts)
    vertsLen = len(curVerts)
    if vertsLen == 0: continue
    # Close the current shape
    allVerts.append(curVerts[0,:])
    separationIdxs.append(idxOffset + vertsLen)
    idxOffset += vertsLen + 1
  # Take away last separator if it exists
  if len(separationIdxs) > 0:
    separationIdxs.pop()
  allVerts = np.vstack(allVerts)
  isfinite = np.ones(len(allVerts), bool)
  isfinite[separationIdxs] = False
  return XYVertices(allVerts, dtype=float), isfinite
  #return XYVertices(dtype=float)


def splitListAtNans(concatVerts:XYVertices):
  """
  Utility for taking a single list of nan-separated region vertices
  and breaking it into several regions with no nans.
  """
  allVerts = []
  nanEntries = np.nonzero(np.isnan(concatVerts[:,0]))[0]
  curIdx = 0
  for nanEntry in nanEntries:
    curVerts = concatVerts[curIdx:nanEntry,:].astype('int')
    allVerts.append(curVerts)
    curIdx = nanEntry+1
  # Account for final grouping of verts
  allVerts.append(concatVerts[curIdx:,:].astype('int'))
  return ComplexXYVertices(allVerts, coerceListElements=True)

def getClippedBbox(arrShape: tuple, bbox: TwoDArr, margin: int):
  """
  Given a bounding box and margin, create a clipped bounding box that does not extend
  past any dimension size from arrShape

  Parameters
  ----------
  arrShape :    2-element tuple
     Refrence array dimensions

  bbox     :    2x2 array
     [minX minY; maxX maxY] bounding box coordinates

  margin   :    int
     Offset from bounding box coords. This will not fully be added to the bounding box
     if the new margin causes coordinates to fall off either end of the reference array shape.
  """
  bbox = bbox.astype(int)
  bbox[0] -= margin
  # Add extra 1 since python slicing stops before last index
  bbox[1] += margin+1
  arrShape = arrShape[:2]
  return np.clip(bbox, 0, arrShape[::-1])

def coerceDfTypes(dataframe: df, constParams: Collection[FRParam]=None):
  """
  Pandas currently has a bug where datatypes are not preserved after update operations.
  Current workaround is to coerce all types to their original values after each operation
  """
  if constParams is None:
    constParams = dataframe.columns
  for field in constParams:
    try:
      dataframe[field] = dataframe[field].astype(type(field.value))
    except TypeError:
      # Coercion isn't possible, nothing to do here
      pass
  return dataframe

def largestList(verts: List[XYVertices]) -> XYVertices:
  maxLenList = []
  for vertList in verts:
    if len(vertList) > len(maxLenList): maxLenList = vertList
  # for vertList in newVerts:
  # vertList += cropOffset[0:2]
  return XYVertices(maxLenList)

def helpTextToRichText(helpText: str, prependText='', postfixText=''):
  # Outside <qt> tags
  if helpText.startswith('<qt>'):
    unwrappedHelpText = helpText[4:-5]
  else:
    unwrappedHelpText = helpText
  if len(prependText) > 0 and len(helpText) > 0 or len(postfixText) > 0:
    prependText += '<br>'
  curText = prependText + unwrappedHelpText
  if len(postfixText) > 0:
    curText += '<br>' + postfixText
  newHelpText = f'<qt>{curText}</qt>'
  return newHelpText


def resolveAuthorName(providedAuthName: Optional[str]) -> Optional[str]:
  authPath = Path(ANN_AUTH_DIR)
  authFile = authPath.joinpath('defaultAuthor.txt')
  if providedAuthName is not None:
    # New default author provided
    with open(authFile.absolute(), 'w') as ofile:
      ofile.write(providedAuthName)
      return providedAuthName
  # Fetch default author
  if not authFile.exists():
    authFile.touch()
  with open(str(authFile), 'r') as ifile:
    lines = ifile.readlines()
    if not lines:
      return None
  return lines[0]

def augmentException(ex: Exception, prependedMsg: str):
  ex.args = (prependedMsg + str(ex),)

def makeUniqueBaseClass(obj: Any):
  """
  Overwrites obj's class to a mixin base class.
  Property objects only work in Python if assigned to the *class* of an object, e.g.
  >>> class b:
  >>>   num = property(lambda self: 4)
  >>> ob = b()
  >>> ob.num # works as expected
  >>> ob.num2 = property(lambda self: 6)
  >>> ob.num2 # Property object at ... -- NOT AS EXPECTED!
  To work around this, simply use <type(ob).num2 = ...>. However, for regisetering properties,
  this means *all* objects of that type will have the properties of all other objects.
  To fix this, a mixin is added to this object and the property is added to the mixin.
  That way, the original object class is not altered, and each separate object will not
  end up sharing the same parameters.
  In summary, this feature enables the assignment
  >>> type(ob).a = property(...)
  without all other `b` objects also receiving this property.
  """
  class mixin(type(obj)): pass
  obj.__class__ = mixin
  return mixin


def pascalCaseToTitle(name: str, addSpaces=True) -> str:
  """
  Helper utility to turn a PascalCase name to a 'Title Case' title
  :param name: camel-cased name
  :param addSpaces: Whether to add spaces in the final result
  :return: Space-separated, properly capitalized version of :param:`Name`
  """
  if not name:
    return name
  if name.startswith('FR'):
    name = name[2:]
  if addSpaces:
    replace = r'\1 \2'
  else:
    replace = r'\1\2'
  name = re.sub(r'(\w)([A-Z])', replace, name)
  name = name.replace('_', ' ')
  return name.title()

def lower_NoSpaces(name: str):
  return name.replace(' ', '').lower()

def safeCallFuncList(fnNames: Collection[str], funcLst: List[Callable],
                     fnArgs: List[Sequence]=None):
  errs = []
  rets = []
  if fnArgs is None:
    fnArgs = [()]*len(fnNames)
  for key, fn, args in zip(fnNames, funcLst, fnArgs):
    try:
      rets.append(fn(*args))
    except Exception as ex:
      errs.append(f'{key}: {ex}')
      rets.append(None)
  return rets, errs

def cornersToFullBoundary(cornerVerts: Union[XYVertices, ComplexXYVertices], sizeLimit: float=np.inf,
                          fillShape: Tuple[int]=None, stackResult=True) -> Union[XYVertices, ComplexXYVertices]:
  """
  From a list of corner vertices, returns a list with one vertex for every border pixel.
  Example:
  >>> cornerVerts = XYVertices([[0,0], [100,0], [100,100],[0,100]])
  >>> cornersToFullBoundary(cornerVerts)
  # [[0,0], [1,0], ..., [100,0], [100,1], ..., [100,100], ..., ..., [0,100]]
  :param cornerVerts: Corners of the represented polygon
  :param sizeLimit: The largest number of pixels from the enclosed area allowed before the full boundary is no
  longer returned. For instance:
    >>> cornerVerts = XYVertices([[0,0], [1000,0], [1000,1000],[0,1000]])
    >>> cornersToFullBoundary(cornerVerts, 10e5)
    will *NOT* return all boundary vertices, since the enclosed area (10e6) is larger than sizeLimit.
  :param fillShape: Size of mask to create. Useful if verts may extend beyond image dimensions
    and should be truncated. If None, no truncation will occur except for negative verts.
  :param stackResult: Whether the result should be ComplexXYVertices (if stackResult is False)
    or a stacked list of exterior verts (if stackResult is True)
  :return: List with one vertex for every border pixel, unless *sizeLimit* is violated.
  """
  if isinstance(cornerVerts, XYVertices):
    cornerVerts = ComplexXYVertices([cornerVerts])
  if fillShape is not None:
    fillShape = tuple(fillShape)
  filledMask = cornerVerts.toMask(fillShape, warnIfTooSmall=False)
  cornerVerts = ComplexXYVertices.fromBwMask(filledMask, simplifyVerts=False)
  if not stackResult:
    return cornerVerts
  cornerVerts = cornerVerts.filledVerts().stack()
  numCornerVerts = len(cornerVerts)
  if numCornerVerts > sizeLimit:
    spacingPerSamp = int(numCornerVerts/sizeLimit)
    cornerVerts = cornerVerts[::spacingPerSamp]
  return cornerVerts


def getCroppedImg(image: NChanImg, verts: np.ndarray, margin: int, *otherBboxes: np.ndarray,
                  coordsAsSlices=False, returnSlices=True) -> (np.ndarray, np.ndarray):
  verts = np.vstack(verts)
  img_np = image
  compCoords = np.vstack([verts.min(0), verts.max(0)])
  if len(otherBboxes) > 0:
    for dim in range(2):
      for ii, cmpFunc in zip(range(2), [min, max]):
        otherCmpVals = [curBbox[ii, dim] for curBbox in otherBboxes]
        compCoords[ii,dim] = cmpFunc(compCoords[ii,dim], *otherCmpVals)
  compCoords = getClippedBbox(img_np.shape, compCoords, margin)
  coordSlices = (slice(compCoords[0,1], compCoords[1,1]),
                 slice(compCoords[0,0],compCoords[1,0]))
  # Verts are x-y, index into image with row-col
  indexer = coordSlices
  if image.ndim > 2:
    indexer += (slice(None),)
  croppedImg = image[indexer]
  if not returnSlices:
    return croppedImg
  if coordsAsSlices:
    return croppedImg, coordSlices
  else:
    return croppedImg, compCoords

def imgCornerVertices(img: NChanImg=None):
  """Returns [x,y] vertices for each corner of the input image"""
  if img is None:
    return XYVertices()
  fullImShape_xy = img.shape[:2][::-1]
  return XYVertices([[0,                   0],
              [0,                   fullImShape_xy[1]-1],
              [fullImShape_xy[0]-1, fullImShape_xy[1]-1],
              [fullImShape_xy[0]-1, 0]
              ])

def resize_pad(img: NChanImg, newSize: Tuple[int, int], interp=cv.INTER_NEAREST, padVal=0):
  """
  Resizes image to the requested size using the specified interpolation method.
  :param img: Image to resize
  :param newSize: New size for image. Since aspect ratio is maintained, the portion of the
    image which couldn't be resized fully is padded with a constant value of `padVal`.
    For instance, if the original image is 5x10 and the requested new size is 10x15, then
    after resizing the image will be 7x15 to preserve aspect ratio. 2 pixels of padding
    will be added on the left and 1 pixel of padding will be added on the right so the final
    output is 10x15.
  :param padVal: Value to pad dimension that couldn't be fully resized
  :param interp: Interpolation method to use during resizing
  :param padVal: Constant value to use during padding
  :return: Resized and padded image
  """
  needsRotate = False
  # Make sure largest dimension is along rows for easier treatment and rotate back after
  newSize = np.array(newSize)
  resizeRatio = min(newSize/np.array(img.shape[:2]))
  paddedImg = cv.resize(img, (0, 0), fx = resizeRatio, fy = resizeRatio, interpolation=interp)
  padDimension = np.argmax(newSize - np.array(paddedImg.shape[:2]))
  if padDimension != 1:
    paddedImg = cv.rotate(paddedImg, cv.ROTATE_90_CLOCKWISE)
    newSize = newSize[::-1]
    needsRotate = True
  # Now pad dimension is guaranteed to be index 1
  padDimension = 1
  padding = (newSize[padDimension] - paddedImg.shape[padDimension]) // 2
  if padding > 0:
    paddedImg = cv.copyMakeBorder(paddedImg, 0, 0, padding, padding, cv.BORDER_CONSTANT, value=padVal)
  # Happens during off-by-one truncated division
  remainingPadding = newSize[padDimension] - paddedImg.shape[padDimension]
  if remainingPadding > 0:
    paddedImg = cv.copyMakeBorder(paddedImg, 0, 0, remainingPadding, 0, cv.BORDER_CONSTANT, value=padVal)
  if needsRotate:
    paddedImg = cv.rotate(paddedImg, cv.ROTATE_90_COUNTERCLOCKWISE)
  return paddedImg

def dynamicDocstring(**kwargs):
  """
  Docstrings must be known at compile time. However this prevents expressions like

  ```
  x = ['dog', 'cat', 'squirrel']
  def a(animal: str):
    \"\"\"
    :param animal: must be one of {x}
    \"\"\"
  ```

  from compiling. This can make some featurs of s3a difficult, like dynamically generating
  limits for a docstring list. `dynamicDocstring` wrapps a docstring and provides kwargs
  for string formatting.
  Retrieved from https://stackoverflow.com/a/10308363/9463643

  :param kwargs: List of kwargs to pass to formatted docstring
  """
  def wrapper(obj):
    obj.__doc__ = obj.__doc__.format(**kwargs)
    return obj
  return wrapper

def resolveYamlDict(cfgFname: FilePath, cfgDict: dict=None):
  if cfgDict is not None:
    cfg = cfgDict
  else:
    cfg = attemptFileLoad(cfgFname)
  return Path(cfgFname), cfg

def serAsFrame(ser: pd.Series):
  return ser.to_frame().T

def attemptFileLoad(fpath: FilePath , openMode='r') -> Union[dict, bytes]:
  with open(fpath, openMode) as ifile:
    loadObj = yaml.load(ifile)
  return loadObj


def getParamChild(param: Parameter, *childPath: Sequence[str], allowCreate=True, groupOpts:dict=None,
                  chOpts: dict=None):
  if groupOpts is None:
    groupOpts = {}
  groupOpts.setdefault('type', 'group')
  while childPath and childPath[0] in param.names:
    param = param.child(childPath[0])
    childPath = childPath[1:]
  # All future children must be created
  if allowCreate:
    for chName in childPath:
      param = param.addChild(dict(name=chName, **groupOpts))
      childPath = childPath[1:]
  elif len(childPath) > 0:
    # Child doesn't exist
    raise KeyError(f'Children {childPath} do not exist in param {param}')
  if chOpts is not None:
    if chOpts['name'] in param.names:
      param = param.child(chOpts['name'])
    else:
      param = param.addChild(chOpts)
  return param


def getAllBases(cls):
  baseClasses = [cls]
  nextClsPtr = 0
  if not isclass(cls):
    return baseClasses
  # Get all bases of bases, too
  while nextClsPtr < len(baseClasses):
    curCls = baseClasses[nextClsPtr]
    curBases = curCls.__bases__
    # Only add base classes that haven't already been added to prevent infinite recursion
    baseClasses.extend([tmpCls for tmpCls in curBases if tmpCls not in baseClasses])
    nextClsPtr += 1
  return baseClasses


def hierarchicalUpdate(curDict: dict, other: dict):
  """Dictionary update that allows nested keys to be updated without deleting the non-updated keys"""
  if other is None:
    return
  for k, v in other.items():
    curVal = curDict.get(k, None)
    if isinstance(curVal, dict) and isinstance(v, dict):
      hierarchicalUpdate(curVal, v)
    elif isinstance(curVal, list) and isinstance(v, list):
      curVal.extend(v)
    else:
      curDict[k] = v

def coerceAnnotation(ann: Any):
  """
  From a function argument annotation, attempts to find a default value matching
  that annotation. E.g. for the function signature
  `def a(param: int)`
  `inspect` is used to find the signature, and `param`'s annotation would be `int`.
  This function would turn that into `int()`, or 0. Attempts are as follows:
    - If the annotation is a string, attempts are made to convert it to a value, e.g.
      `def a(param: 'int')` will follow the flow of `def a(param: int).Note this
       will fail if the annotation is for a class not in scope upon evaluation.
    - If direct conversion works as described in the docstring, the value is returned.

       * If this fails due to *TypeError*, the following cases are examined:
         * typing.Union (flow is followed for each union type until non-None is
         returned or all types have been examined)
         * typing.List, Tuple, Dict, <builtim>: That builtin is returned

     - All other cases result in *None* being returned.
  """
  if isinstance(ann, str):
    ann = literal_eval(ann)
  try:
    return ann()
  except TypeError:
    # Check for typing types
    if getattr(ann, '__module__', None) == t.__name__:
      if ann.__origin__ is t.Union:
        ann: t.Union
        for arg in ann.__args__:
          ret = coerceAnnotation(arg)
          if ret is not None:
            return ret
      else:
        # Check for typing instances of builtins
        if hasattr(ann, '_name'):
          try:
            return literal_eval(ann._name.lower())
          except Exception as ex:
            pass
  return None

@contextmanager
def monkeyPatch(obj, toChange: str, newVal):
  oldVal = getattr(obj, toChange, None)
  setattr(obj, toChange, newVal)
  yield
  if oldVal is None:
    delattr(obj, toChange)
  else:
    setattr(obj, toChange, oldVal)