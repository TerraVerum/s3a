from __future__ import annotations

import inspect
import warnings
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Collection, List, Sequence, Tuple, Union

import cv2 as cv
import numpy as np
import pandas as pd
import pyqtgraph as pg
from pkg_resources import parse_version
from pyqtgraph.parametertree.interactive import InteractiveFunction
from skimage import __version__ as _skimage_version, io, transform as trans
from skimage.exposure import exposure
from utilitys import PrjParam, fns

# Needs to be visible outside this file
from utilitys.fns import hierarchicalUpdate  # noqa
from utilitys.typeoverloads import FilePath

from .constants import PRJ_ENUMS
from .structures import BlackWhiteImg, ComplexXYVertices, NChanImg, TwoDArr, XYVertices

_coordType = Union[np.ndarray, Tuple[slice, slice]]
USE_MULTICHANNEL_KWARG = parse_version(_skimage_version) < parse_version("0.19.0")


def stackedVerticesPlusConnections(
    vertices: ComplexXYVertices,
) -> Tuple[XYVertices, np.ndarray]:
    """
    Utility for concatenating all vertices within a list while recording where
    separations occurred
    """
    allVerts = [np.zeros((0, 2))]
    separationIdxs = []
    idxOffset = 0
    for curVerts in vertices:
        allVerts.append(curVerts)
        vertsLen = len(curVerts)
        if vertsLen == 0:
            continue
        # Close the current shape
        allVerts.append(curVerts[0, :])
        separationIdxs.append(idxOffset + vertsLen)
        idxOffset += vertsLen + 1
    # Take away last separator if it exists
    if len(separationIdxs) > 0:
        separationIdxs.pop()
    allVerts = np.vstack(allVerts)
    isfinite = np.ones(len(allVerts), bool)
    isfinite[separationIdxs] = False
    return XYVertices(allVerts, dtype=float), isfinite


def getClippedBoundingBox(shape: tuple, boundingBox: TwoDArr, margin: int):
    """
    Given a bounding box and margin, create a clipped bounding box that does not extend
    past any dimension size from arrShape

    Parameters
    ----------
    shape : 2-element tuple
        Refrence array dimensions
    boundingBox : 2x2 array
        [minX minY; maxX maxY] bounding box coordinates
    margin : int
        Offset from bounding box coords. This will not fully be added to the bounding
        box if the new margin causes coordinates to fall off either end of the
        reference array shape.
    """
    boundingBox = boundingBox.astype(int)
    boundingBox[0] -= margin
    # Add extra 1 since python slicing stops before last index
    boundingBox[1] += margin + 1
    shape = shape[:2]
    return np.clip(boundingBox, 0, shape[::-1])


def coerceDfTypes(dataframe: pd.DataFrame, constParams: Collection[PrjParam] = None):
    """
    Pandas currently has a bug where datatypes are not preserved after update
    operations. Current workaround is to coerce all types to their original values
    after each operation
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
        if len(vertList) > len(maxLenList):
            maxLenList = vertList
    # for vertices in newVertices:
    # vertices += cropOffset[0:2]
    return XYVertices(maxLenList)


def augmentException(ex: Exception, prependedMsg: str):
    exMsg = str(ex)
    ex.args = (prependedMsg + exMsg,)


def lowerNoSpaces(name: str):
    return name.replace(" ", "").lower()


def safeCallFunctionList(
    fnNames: Collection[str], funcLst: List[Callable], fnArgs: List[Sequence] = None
):
    errs = []
    rets = []
    if fnArgs is None:
        fnArgs = [()] * len(fnNames)
    for key, fn, args in zip(fnNames, funcLst, fnArgs):
        curRet, curErr = safeCallFunction(key, fn, *args)
        rets.append(curRet)
        if curErr:
            errs.append(curErr)
    return rets, errs


def safeCallFunction(fnName: str, func: Callable, *fnArgs):
    ret = err = None
    try:
        ret = func(*fnArgs)
    except Exception as ex:
        err = f"{fnName}: {ex}"
    return ret, err


def _maybeBgrToRgb(image: np.ndarray):
    """Treats 3/4-channel images as BGR/BGRA for opencv saving/reading"""
    if image.ndim > 2:
        # if image.shape[0] == 1:
        #   image = image[...,0]
        if image.shape[2] >= 3:
            lastAx = np.arange(image.shape[2], dtype="int")
            # Swap B & R
            lastAx[[0, 2]] = [2, 0]
            image = image[..., lastAx]
    return image


def cvImsaveRgb(fname: FilePath, image: np.ndarray, *args, errOk=False, **kwargs):
    image = _maybeBgrToRgb(image)
    try:
        cv.imwrite(str(fname), image, *args, **kwargs)
    except cv.error:
        if not errOk:
            raise
        # Dtype incompatible
        io.imsave(fname, image)


def cvImreadRgb(fname: FilePath, *args, **kwargs):
    image = cv.imread(str(fname), *args, **kwargs)
    return _maybeBgrToRgb(image)


def tryCvResize(
    image: NChanImg, newSize: Union[tuple, float], asRatio=True, interp=cv.INTER_CUBIC
):
    """
    Uses ``cv.resize`` where posisble, but if dtypes prevent this, it falls back to
    skimage.transform.rescale/resize

    Parameters
    ----------
    image
        Image to resize
    newSize
        Either ratio to scale each axis or new image size (x, y -- not row, col)
    asRatio
        Whether to interpret `newSize` as a ratio or new image dimensions
    interp
        Interpolation to use, if cv.resize is available for the given dtype
    """
    if asRatio:
        if not isinstance(newSize, tuple):
            newSize = (newSize, newSize)
        args = dict(dsize=(0, 0), fx=newSize[0], fy=newSize[1])
    else:
        args = dict(dsize=newSize)
    try:
        image = cv.resize(image, **args, interpolation=interp)
    except (TypeError, cv.error):
        oldRange = (image.min(), image.max())
        if asRatio:
            kwarg = dict(channel_axis=2 if image.ndim > 2 else None)
            if USE_MULTICHANNEL_KWARG:
                kwarg = dict(multichannel=kwarg["channel_axis"] is not None)
            rescaled = trans.rescale(image, newSize, **kwarg)
        else:
            rescaled = trans.resize(image, newSize[::-1])
        image = exposure.rescale_intensity(rescaled, out_range=oldRange).astype(
            image.dtype
        )
    return image


def cornersToFullBoundary(cornerVerts: XYVertices, sizeLimit=0) -> XYVertices:
    """
    From a list of corner vertices, returns an array with one vertex for every border
    pixel.

    Parameters
    ----------
    cornerVerts
        Corners of the represented polygon
    sizeLimit
        If > 0, every nth pixel from the border will be used such that ``sizeLimit``
        total points are returned

    Returns
    -------
    XYVertices
        List with one vertex for every border pixel, unless ``sizeLimit`` is violated.
    """
    # Credit: https://stackoverflow.com/a/70664846/9463643
    # Cumulative Euclidean distance between successive polygon points.
    # This will be the "x" for interpolation
    # Ensure shape is connected
    vertices = np.r_[cornerVerts, cornerVerts[[0]]]
    d = np.cumsum(np.r_[0, np.sqrt((np.diff(vertices, axis=0) ** 2).sum(axis=1))])

    # Edge case: vertices don't enclose any usable area, return without changes
    # otherwise all vertices will disappear
    if np.max(d) == 0:
        return cornerVerts

    # get linearly spaced points along the cumulative Euclidean distance
    if sizeLimit > 0:
        distSampled = np.linspace(0, d.max(), int(sizeLimit))
    else:
        distSampled = np.arange(0, d.max())

    # interpolate x and y coordinates
    vertsInterped = np.c_[
        np.interp(distSampled, d, vertices[:, 0]),
        np.interp(distSampled, d, vertices[:, 1]),
    ]

    return XYVertices(vertsInterped)


def getCroppedImage(
    image: NChanImg,
    vertices: np.ndarray,
    margin=0,
    returnSlices=False,
    returnBoundingBox=True,
) -> tuple[np.ndarray, _coordType] | np.ndarray:
    """
    Crops an image according to the specified vertices such that the returned image
    does not extend past vertices plus margin (including other bboxes if specified).

    Parameters
    ----------
    image
        Image to crop
    vertices
        Vertices of the polygon to crop to
    margin
        Margin to add to the top, bottom, left, and right of the polygon formed
        by ``vertices``
    returnSlices
        Whether to return the slices used to crop the image. In the format
        ``slice(yMin, yMax), slice(xMin, xMax)``.
    returnBoundingBox
        Whether to return the bounding box of the cropped image. In the format
        ``[[xmin, ymin], [xmax, ymax]]``.
    """
    vertices = np.vstack(vertices)
    img_np = image
    compCoords = np.vstack([vertices.min(0), vertices.max(0)])
    compCoords = getClippedBoundingBox(img_np.shape, compCoords, margin)
    coordSlices = (
        slice(compCoords[0, 1], compCoords[1, 1]),
        slice(compCoords[0, 0], compCoords[1, 0]),
    )
    # Verts are x-y, index into image with row-col
    indexer = coordSlices
    if image.ndim > 2:
        indexer += (slice(None),)
    toReturn = [image[indexer]]
    if returnSlices:
        toReturn.append(coordSlices)
    if returnBoundingBox:
        toReturn.append(compCoords)
    return tuple(toReturn) if len(toReturn) > 1 else toReturn[0]


def imageCornerVertices(image: NChanImg = None) -> XYVertices:
    """Returns [x,y] vertices for each corner of the input image"""
    if image is None:
        return XYVertices()
    fullImShape_xy = image.shape[:2][::-1]
    return XYVertices(
        [
            [0, 0],
            [0, fullImShape_xy[1] - 1],
            [fullImShape_xy[0] - 1, fullImShape_xy[1] - 1],
            [fullImShape_xy[0] - 1, 0],
        ]
    )


def showMaskDifference(oldMask: BlackWhiteImg, newMask: BlackWhiteImg):
    infoMask = np.tile(oldMask[..., None].astype("uint8") * 255, (1, 1, 3))
    # Was there, now it's not -- color red
    infoMask[oldMask & ~newMask, :] = [255, 0, 0]
    # Wasn't there, now it is -- color green
    infoMask[~oldMask & newMask, :] = [0, 255, 0]
    return infoMask


# A different `maxsize` doesn't change whether two dicts have equal values,
# which is why dict.__eq__ holds here.
class MaxSizeDict(dict):  # lgtm [py/missing-equals]
    """
    Poor man's LRU dict, since I don't yet feel like including another pypi dependency
    Rather than evicting the least recently used, it evicts the oldest set value.
    Allows simpler implementation since no logic is needed to track last accesses. Use
    something more effective than this if true LRU behavior is required.
    """

    def __init__(self, *args, maxsize: int = np.inf, **kwargs):
        super().__init__(*args, **kwargs)
        self.maxsize = maxsize

    def __setitem__(self, key, value):
        # Perform set before pop in case errors arise
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            # Evict oldest accessed entry
            self.pop(next(iter(self.keys())))


def _getPointAngles(pts):
    midpt = np.mean(pts, 0)
    relPosPts = (pts - midpt).view(np.ndarray)
    return np.arctan2(*relPosPts.T[::-1])


def orderContourPoints(pts: XYVertices, ccw=True):
    """
    Only guaranteed to work for convex hulls, i.e. shapes not intersecting themselves.
    Orderes an arbitrary list of coordinates into one that works well line plotting,
    i.e. doesn't show excessive self-crossing when plotting
    """
    if len(pts) < 3:
        return pts
    angles = _getPointAngles(pts)
    ptOrder = np.argsort(angles)
    if not ccw:
        ptOrder = ptOrder[::-1]
    return pts[ptOrder]


def symbolFromVertices(vertices: Union[ComplexXYVertices, XYVertices, np.ndarray]):
    if isinstance(vertices, ComplexXYVertices):
        concatRegion, isfinite = stackedVerticesPlusConnections(vertices)
    else:
        concatRegion, isfinite = vertices, np.all(np.isfinite(vertices), axis=1)
        # Qt doesn't like subclassed ndarrays
        concatRegion = concatRegion.view(np.ndarray)
    if not len(concatRegion):
        boundLoc = np.array([[0, 0]])
    else:
        boundLoc = np.nanmin(concatRegion, 0, keepdims=True)
    useVerts = concatRegion - boundLoc + 0.5
    # pyqtgraph 0.12.2 errs on an empty symbol
    # https://github.com/pyqtgraph/pyqtgraph/issues/1888
    if not len(isfinite):
        isfinite = "all"
    return pg.arrayToQPath(*useVerts.T, connect=isfinite), boundLoc


# Credit: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/  # noqa
def boundingBoxIou(boxA, boxB):
    """
    determine the (x, y)-coordinates of the intersection rectangle. Both boxes are
    formatted [[xmin, ymin], [xmax, ymax]]
    """
    boxA = boxA.ravel()
    boxB = boxB.ravel()
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def getTopLeftCoordinate(verts: XYVertices):
    """
    Finds the coordinate in the vertices list that is closest to the origin. `vertices`
    must have length > 0.
    """
    # Early exit condition
    if verts.ndim < 2:
        # One point
        return verts
    allMin = verts.min(0)
    closestPtIdx = np.argmin(np.sum(np.abs(verts - allMin), 1))
    return verts[closestPtIdx]


def imagePathToHtml(imgPath: FilePath, width=None):
    outStr = f'<img src="{imgPath}"'
    if width is not None:
        outStr += f' width="{width}px"'
    outStr += ">"
    return outStr


class DirectoryDict(MaxSizeDict):
    """
    Used to shim the API between file-system and programmatically generated content. If
    a directory is passed, files are read and cached when a name is passed. Otherwise,
    treat as a normal dict of things. For instance, a directory of png images can be
    accessed through this data structure as ddict['x.png'] which will load the image.
    Next time 'x.png' is accessed, it will be instantaneous due to caching. Similarly,
    `ddict` could be given an initial dict if contents are not directory-backed
    """

    _UNSET = object()
    # Define readFunc and allowAbsolute here to suppress PyCharm warnings in __init__
    readFunc: Any
    allowAbsolute: Any

    def __init__(
        self,
        initData: Union[FilePath, dict, "DirectoryDict"] = None,
        readFunc: Callable[[str], Any] = None,
        allowAbsolute=False,
        cacheOnRead=True,
        **kwargs,
    ):
        """
        Parameters
        ----------
        initData
            Either startup dict or backing directory path. If a DirectoryDict is
            passed, its attribute will be used instead of the value passed for
            allowAbsolute. Its readFunc will be used if the passed readFunc is *None*
        readFunc
            Function used to read files from the directory, i.e. ``io.imread``,
            ``attemptFileLoad``, etc. Must accept the name of the file to read
        allowAbsolute
            Whether to allow reading absolute paths
        **kwargs
            Passed to super constructor
        """
        self.fileDir = None
        if isinstance(initData, FilePath.__args__):
            self.fileDir = Path(initData)
            super().__init__(**kwargs)
        else:
            if isinstance(initData, DirectoryDict):
                readFunc = readFunc or initData.readFunc
                allowAbsolute = initData.allowAbsolute
                self.fileDir = initData.fileDir
            if initData is None:
                initData = {}
            super().__init__(initData, **kwargs)
        self.readFunc = readFunc
        self.allowAbsolute = allowAbsolute
        self.cacheReads = cacheOnRead

    def __missing__(self, key):
        key = str(key)
        exists = super().get(key, self._UNSET)
        if exists is not self._UNSET:
            return exists
        if self.fileDir is None:
            raise KeyError(
                f'"{key}" is not in dict and no backing file system was provided'
            )
        pathKey = Path(key)
        isAbsolute = pathKey.is_absolute()
        if not self.allowAbsolute and isAbsolute:
            raise KeyError(f"Directory paths must be relative, received {key}")
        testPath = pathKey if isAbsolute else self.fileDir / key
        candidates = list(testPath.parent.glob(testPath.name))
        if len(candidates) != 1:
            grammar = ": " if len(candidates) else ""
            raise KeyError(
                f'"{key}" corresponds to {len(candidates)} files{grammar}'
                f'{", ".join(c.name for c in candidates)} '
            )
        else:
            file = candidates[0]
            ret = self.readFunc(str(file))
            if self.cacheReads:
                self[key] = ret
        return ret

    def get(self, key, default=None):
        ret = super().get(key, self._UNSET)
        if ret is self._UNSET:
            # See if directory has this data
            try:
                ret = self[key]
            except KeyError:
                return default
        return ret

    def __eq__(self, other):
        contentsEq = super().__eq__(other)
        if not contentsEq or not isinstance(other, DirectoryDict):
            return contentsEq
        # Since additional keys can come from the directory (in theory determining
        # dict contents), check for equality on those attributes
        return (
            self.fileDir.resolve() == other.fileDir.resolve()
            # TODO: Should readfuncs be allowed to differ? This would require some sort
            #   of type-checking on its output
            and self.readFunc == other.readFunc
        )

    def __ne__(self, other):
        # since dict is a base class, __ne__ doesn't work out of the box unless it is
        # also expliticly defined
        return not self.__eq__(other)


def deprecateKwargs(warningType=DeprecationWarning, **oldToNewNameMapping):
    def deco(func):
        @wraps(func)
        def inner(*args, **kwargs):
            usedDeprecated = set(oldToNewNameMapping) & set(kwargs)
            if usedDeprecated:
                grammar = "is" if len(usedDeprecated) == 1 else "are"
                replacements = {
                    k: oldToNewNameMapping[k]
                    for k in usedDeprecated
                    if oldToNewNameMapping[k] is not None
                }
                msg = (
                    f'{", ".join(usedDeprecated)} {grammar} deprecated and will be '
                    f"removed in a future release."
                )
                if replacements:
                    for orig, replace in replacements.items():
                        kwargs[replace] = kwargs[orig]
                        del kwargs[orig]
                    msg += f" Use the following replacement guide: {replacements}"
                warnings.warn(msg, warningType, stacklevel=3)
            return func(*args, **kwargs)

        return inner

    return deco


def _indexUsingPad(image, tblrPadding):
    """
    Extracts an inner portion of an image accounting for top/bottom/left/right padding
    tuple
    """
    imshape = image.shape[:2]
    rows = slice(tblrPadding[0], imshape[0] - tblrPadding[1])
    cols = slice(tblrPadding[2], imshape[1] - tblrPadding[3])
    return image[rows, cols, ...]


def subImageFromVertices(
    image,
    vertices: ComplexXYVertices | np.ndarray,
    margin=0,
    shape=None,
    returnBoundingBox=False,
    returnStats=False,
    allowTranspose=False,
    rotationDegrees=0,
    **kwargs,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """
    Extracts a region from an image cropped to the area of `vertices`. Unlike
    `getCroppedImage`, this allows a constant-sized output shape, rotation, and other
    features.

    Parameters
    ----------
    image
        Full-sized image from which to extract a subregion
    vertices
        Nx2 array of x-y vertices that determine the extracted region
    margin
        Margin in pixels to add to the [x, y] shapes of vertices. Can be a scalar or
        [x,y]. If float values are specified, the margin is interpreted as a percentage
        (1.0 = 100% margin) of the [x,y] sizes.
    shape
        (rows, cols) output shape
    returnBoundingBox
        Whether to return a bounding box array of [[minx, miny], [maxx, maxy]]
        coordinates where this subimage fits. With no scaling, rotation, padding,
        etc. this is the same region as the vertices bounding box
    returnStats
        Whether to return a dict of reshaping stats that can be passed to
        ``inverseSubImage``
    allowTranspose
        If ``True``, the image can be rotated 90 degrees if it results in less padding
        to reach the desired shape
    rotationDegrees
        Clockwise rotation to apply to the extracted image
    **kwargs
        Passed to ``cv.warpAffine``
    """
    if shape is not None:
        shape = np.asarray(shape[:2])

    if isinstance(vertices, ComplexXYVertices):
        vertices = vertices.stack()
    else:
        vertices = vertices.copy()
    if np.isscalar(margin):
        margin = [margin, margin]
    margin = np.asarray(margin)
    if np.issubdtype(margin.dtype, np.float_):
        margin = (margin * vertices.ptp(0)).astype(int)

    for ax in 0, 1:
        vertices[vertices[:, ax].argmin()] -= margin
        vertices[vertices[:, ax].argmax()] += margin

    initialBbox = coordsToBox(vertices)
    transformedPts, rotationDegrees = _getRotationStats(
        vertices, rotationDegrees, shape, allowTranspose
    )
    if shape is None:
        shape = (transformedPts.ptp(0)[::-1] + 1).astype(int)
    transformedBbox = coordsToBox(transformedPts)
    # Handle half-coordinates by preserving both ends of the spectrum
    newXYShape = transformedBbox.ptp(0).astype(int)
    transformedBbox[0] = np.floor(transformedBbox[0])
    transformedBbox[1] = np.ceil(transformedBbox[1])

    # outputShape is row-col, intialShape is xy
    xyRatios = shape[::-1] / newXYShape
    padAx = np.argmax(xyRatios)
    padAmt = np.round((xyRatios.max() / xyRatios.min() - 1) * newXYShape[padAx])
    # left/right could be top/bottom, but the concept is the same either way
    leftPad = rightPad = padAmt / 2

    transformedBbox[0, padAx] -= leftPad
    transformedBbox[1, padAx] += rightPad

    # In order for rotation not to clip any image pixels, make sure to capture a
    # bounding box big enough to prevent border-filling where possible
    if not np.isclose(rotationDegrees, 0):
        rotatedPoints = cv.boxPoints(
            (transformedBbox.mean(0), transformedBbox.ptp(0), -rotationDegrees)
        )
        # rotatedPoints = initialBbox
        totalBbox = np.r_[initialBbox, rotatedPoints, transformedBbox]
    else:
        totalBbox = np.r_[initialBbox, transformedBbox]

    subImage, stats = _getAffineSubregion(
        image, transformedBbox, totalBbox, shape, rotationDegrees, **kwargs
    )
    stats["initialBbox"] = initialBbox

    ret = [subImage]
    if returnBoundingBox:
        ret.append(transformedBbox)
    if returnStats:
        ret.append(stats)
    if len(ret) == 1:
        return ret[0]
    return tuple(ret)


def _getRotationStats(
    verts: ComplexXYVertices | np.ndarray,
    rotationDeg: float | Any,
    outputShape: np.ndarray = None,
    allowTranspose=False,
):
    if isinstance(verts, ComplexXYVertices):
        verts = verts.stack()
    center, width_height, trueRotationDeg = cv.minAreaRect(verts)

    # Box calculation behaves very poorly with small-area vertices
    width_height = np.clip(width_height, 1, np.inf)
    if abs(trueRotationDeg - 90) < abs(trueRotationDeg):
        # Rotations close to 90 are the same as close to 0 but with flipped width/height
        trueRotationDeg -= 90
        width_height = width_height[::-1]

    if rotationDeg is PRJ_ENUMS.ROTATION_OPTIMAL:
        # Use the rotation that most squarely aligns the component
        rotationDeg = trueRotationDeg
        allowTranspose = True

    points = cv.boxPoints((center, width_height, rotationDeg - trueRotationDeg))
    if (
        allowTranspose
        and outputShape is not None
        and _shapeTransposeNeeded(points.ptp(0)[::-1], outputShape)
    ):
        # Redo the rotation with an extra 90 degrees to swap width and height
        # Use whichever re-orientation leads closer to a 0 degree angle
        newRots = [rotationDeg + 90, rotationDeg - 90, rotationDeg]
        newIdx = np.argmin(np.abs(newRots) % 360)
        rotationDeg = newRots[newIdx]
        # Case where width/height should be swapped, as done to trueRotDeg
        if newIdx == 2:
            width_height = width_height[::-1]
        points = cv.boxPoints((center, width_height, rotationDeg - trueRotationDeg))

    return points, rotationDeg


def _shapeTransposeNeeded(inputShape, outputShape):
    """
    Determines whether the input shape would better fit to the output shope if it was
    transposed. This is evaluated based on whether it results in a maintained aspect
    ratio during resizing
    """
    ratios = outputShape / inputShape
    # Choose whichever orientation leads to the closest ratio to the original size
    transposedRatio = outputShape / inputShape[::-1]
    return np.abs(1 - transposedRatio.max() / transposedRatio.min()) < np.abs(
        1 - ratios.max() / ratios.min()
    )


def _getAffineSubregion(
    image,
    transformedBbox: np.ndarray,
    totalBbox: np.ndarray,
    outputShape,
    rotationDeg=0.0,
    padBorderOpts: dict = None,
    **affineKwargs,
):
    totalBbox = coordsToBox(totalBbox, addOneToMax=False)
    # It's common for rotation angles to cut off fractions of a pixel during warping.
    # This is easily resolved by adding just a few more pixels to the total box
    totalBbox[0] -= 1
    totalBbox[1] += 1
    hasRotation = not np.isclose(rotationDeg, 0)
    xyImageShape = image.shape[:2][::-1]
    # It's possible for mins and maxs to be outside image regions
    underoverPadding = np.zeros_like(totalBbox, dtype=int)
    idx = totalBbox[0] < 0
    underoverPadding[0, idx] = -totalBbox[0, idx]
    idx = totalBbox[1] > xyImageShape
    underoverPadding[1, idx] = (totalBbox[1] - xyImageShape)[idx]
    subImageBbox = totalBbox.astype(int)
    normedTransformedBbox = (transformedBbox - subImageBbox[0]).astype(int)
    totalBbox = np.clip(totalBbox, 0, xyImageShape).astype(int)

    mins = totalBbox.min(0)
    maxs = totalBbox.max(0)

    subImage = image[mins[1] : maxs[1], mins[0] : maxs[0], ...]
    if np.any(underoverPadding):
        padBorderOpts = padBorderOpts or dict(value=0, borderType=cv.BORDER_CONSTANT)
        subImage = cv.copyMakeBorder(
            subImage,
            underoverPadding[0, 1],
            underoverPadding[1, 1],
            underoverPadding[0, 0],
            underoverPadding[1, 0],
            **padBorderOpts,
        )
    inter = affineKwargs.pop("interpolation", cv.INTER_NEAREST)
    if hasRotation:
        midpoint = transformedBbox.mean(0) - subImageBbox[0]
        M = cv.getRotationMatrix2D(midpoint, rotationDeg, 1)
        rotated = cv.warpAffine(
            subImage, M, subImage.shape[:2][::-1], flags=inter, **affineKwargs
        )
    else:
        rotated = subImage
    offset = normedTransformedBbox[0]
    assert np.all(offset >= 0)
    assert np.all(normedTransformedBbox[1] <= rotated.shape[:2][::-1])
    # plt.imshow(rotated)
    toRescale = rotated[
        normedTransformedBbox[0, 1] : normedTransformedBbox[1, 1],
        normedTransformedBbox[0, 0] : normedTransformedBbox[1, 0],
        ...,
    ]
    stats = dict(
        subImageBbox=subImageBbox,
        normedTransformedBbox=normedTransformedBbox,
        rotation=rotationDeg,
        interpolation=inter,
        **affineKwargs,
    )
    return cv.resize(toRescale, outputShape[::-1], interpolation=inter), stats


def inverseSubImage(subImage, stats, finalBbox: np.ndarray = None):
    """
    From a subImage after a call from `subImageFromVerts`, turns it back into a
    regularly size, de-rotated version

    Parameters
    ----------
    subImage
        Image to resize and re-rotate
    stats
        dict of stats coming from the `subImageFromVerts` call
    finalBbox
        If provided, this is the region from within the inverted subImage to extract
    """
    subBbox = stats["subImageBbox"]
    transBbox = stats["normedTransformedBbox"]
    preResizedShape = transBbox.ptp(0)
    unresized = cv.resize(
        subImage, preResizedShape, interpolation=stats.get("interpolation")
    )
    # De-rotate around the same center as the rotation occurred, accounting for an
    # offset from subindexing (transBbox[0])
    unrotated = trans.rotate(
        unresized, -stats["rotation"], resize=True, preserve_range=True
    ).astype(subImage.dtype)
    if finalBbox is None and "initialBbox" not in stats:
        return unrotated
    elif finalBbox is None:
        finalBbox = stats["initialBbox"]
    idxOffset = np.clip(finalBbox[0] - (subBbox[0]), 0, np.inf).astype(int)
    outputSizeXY = finalBbox.ptp(0)
    out = unrotated[
        idxOffset[1] : idxOffset[1] + outputSizeXY[1],
        idxOffset[0] : idxOffset[0] + outputSizeXY[0],
        ...,
    ]
    return out


def coordsToBox(coords: np.ndarray, addOneToMax=True):
    ret = np.r_[coords.min(0, keepdims=True), coords.max(0, keepdims=True)]
    if addOneToMax:
        ret[1] += 1
    return ret


def toDictGen(df: pd.DataFrame, index=False):
    """
    pandas to_dict() keeps all rows in memory at once. This method is similar, but is a
    generator version only yielding one row at a time.
    """
    cols = df.columns.to_list()
    # Define out here to avoid if-statement in every evaluation
    if index:
        idx = df.index.to_list()
        for ii, row in enumerate(df.itertuples(index=False)):
            yield idx[ii], dict(zip(cols, row))
    else:
        for row in df.itertuples(index=False):
            yield dict(zip(cols, row))


def toHtmlWithStyle(
    df: pd.DataFrame, file: FilePath = None, style: str = None, **exportArgs
):
    """
    `to_html` doesn't allow any injection of style attributes, etc. This is an
    intermediate step which collects the exported dataframe data, prepends <style> tags
    with the inserted style string, and completes the export. If no style is provided,
    `to_html` is called normally.
    """
    if not style:
        return df.to_html(file, **exportArgs)
    htmlDf = df.to_html(None, **exportArgs)
    style = inspect.cleandoc(
        f"""
    <style>
    {style}
    </style>
    """
    )
    htmlDf = f"{style}\n{htmlDf}"
    if file is not None:
        with open(file, "w") as ofile:
            ofile.write(htmlDf)
    else:
        return htmlDf


def _convertImage(file, ext="png", replace=True):
    if file.suffix[1:] == ext:
        return
    try:
        cv.imwrite(str(file.with_suffix(f".{ext}")), cv.imread(str(file)))
        if replace:
            file.unlink()
    except Exception as ex:
        return file, ex


def convertImages(
    globstr: str = "*.*", ext="png", replace=True, folder: FilePath = None
):
    if folder is None:
        folder = Path()
    folder = Path(folder)
    files = list(folder.glob(globstr))
    ret = fns.mprocApply(
        _convertImage, files, "Converting Files", extraArgs=(ext, replace)
    )
    errs = [f"{r[0].name}: {r[1]}" for r in ret if r is not None]
    if errs:
        print(f"Conversion errors occurred in the following files:\n" + "\n".join(errs))


def getObjectsDefinedInSelfModule(moduleVars, moduleName, ignorePrefix="_"):
    # Prepopulate to avoid "dictionary changed on iteration"
    _iterVars = list(moduleVars.items())
    out = []
    for name, obj in _iterVars:
        if name.startswith(ignorePrefix):
            continue
        if hasattr(obj, "__module__") and obj.__module__ == moduleName:
            out.append(name)
    return out


# Credit: https://github.com/biolab/orange3/blob/master/Orange/misc/cache.py
# Permalink to copy date:
# https://github.com/biolab/orange3/blob/0f5025a74a32a5b69ebf67e555131cc5b540143d/Orange/misc/cache.py  # noqa
# Slightly modified to use equality instead of identity checks
def simpleCache(func):
    """Cache with size 1."""
    # populate the cache with a dummy value to guarantee the first run succeeds
    last_args = (object(),)
    last_kwargs = {object(): object()}
    last_result = None

    @wraps(func)
    def _cached(*args, **kwargs):
        nonlocal last_args, last_kwargs, last_result
        if (
            len(last_args) != len(args)
            or not all(pg.eq(x, y) for x, y in zip(args, last_args))
            or not pg.eq(last_kwargs, kwargs)
        ):
            last_result = func(*args, **kwargs)
            last_args, last_kwargs = args, kwargs
        return last_result

    return _cached


class ClassInteractiveFunction(InteractiveFunction):
    def __get__(self, instance, owner=None):
        if instance is None:
            self.extra.pop("self", None)
            return self
        self.extra["self"] = instance
        return self
