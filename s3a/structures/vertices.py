from __future__ import annotations

from ast import literal_eval
from typing import List, Sequence, TypeVar, Union
from warnings import warn

import cv2 as cv
import numpy as np
import pandas as pd

from .typeoverloads import BlackWhiteImg, NChanImg

T = TypeVar("T")
__all__ = ["XYVertices", "ComplexXYVertices", "S3AVertsAccessor"]


class XYVertices(np.ndarray):
    connected = True

    def __new__(
        cls,
        inputArr: Union[list, np.ndarray, tuple] = None,
        connected=True,
        dtype=int,
        **kwargs,
    ):
        # Default to integer type if not specified, since this is how pixel coordinates
        # will be represented anyway See numpy docs on subclassing ndarray
        if inputArr is None:
            inputArr = np.zeros((0, 2))
        arr = np.asarray(inputArr, dtype=dtype, **kwargs).view(cls)
        arr.connected = connected
        return arr

    def __array_finalize__(self, obj):
        shape = self.shape
        shapeLen = len(shape)
        # indicates point, so the one dimension must have only 2 elements
        if 1 < shapeLen < 2 and shape[0] != 2:
            raise ValueError(
                f"A one-dimensional vertex array must be shape (2,)."
                f" Receieved array of shape {shape}"
            )
        elif shapeLen > 2 or shapeLen > 1 and shape[1] != 2:
            raise ValueError(f"Vertex list must be Nx2. Received shape {shape}.")
        if obj is None:
            return
        self.connected = getattr(obj, "connected", True)

    @property
    def empty(self):
        return len(self) == 0

    def serialize(self):
        return str(self.tolist())

    @classmethod
    def deserialize(cls, stringValue: str):
        out = cls(literal_eval(stringValue))
        if out.size == 0:
            # Make sure size is 0x2
            return cls()
        return out

    # def asPoint(self):
    #   if self.size == 2:
    #     return self.reshape(-1)
    #   # Reaching here means the user requested vertices as point when
    #   # more than one point is in the list
    #   raise ValueError(f'asPoint() can only be called when one vertex is in'
    #                             f' the vertex list. Currently has shape {self.shape}')

    # def asRowCol(self):
    #   return np.fliplr(self)

    @property
    def x(self):
        # Copy to array first so dimensionality checks are no longer required
        return np.array(self).reshape(-1, 2)[:, [0]]

    @x.setter
    def x(self, newX):
        self.reshape(-1, 2)[:, 0] = newX

    @property
    def y(self):
        return np.array(self).reshape(-1, 2)[:, [1]]

    @y.setter
    def y(self, newY):
        self.reshape(-1, 2)[:, 1] = newY

    @property
    def rows(self):
        return self.y

    @rows.setter
    def rows(self, newRows):
        self.y = newRows

    @property
    def columns(self):
        return self.x

    @columns.setter
    def columns(self, newColumns):
        self.x = newColumns


class ComplexXYVertices(list):
    """
    Allows holes in the component shape. Subclassing ndarray instead of list allows
    primitive algebraic ops on the list contents (e.g. subtracting/adding offset).
    Since normal usage doesn't typically require a mutable structure, the loss is
    minimal.
    """

    hierarchy = np.ones((0, 4), dtype=int)
    """See cv.findContours for hierarchy explanation. Used in cv.RETR_CCOMP mode."""

    def __init__(
        self,
        verticesArray: Union[List[XYVertices], np.ndarray, Sequence] = None,
        hierarchy: np.ndarray = None,
        coerceListElements=False,
    ):
        if verticesArray is None:
            verticesArray = []
        numInpts = len(verticesArray)
        if coerceListElements:
            verticesArray = [XYVertices(el) for el in verticesArray]
        if hierarchy is None:
            hierarchy = np.ones((numInpts, 4), dtype=int) * -1
        super().__init__(verticesArray)
        # No hierarchy required unless list is longer than length 1
        self.hierarchy = hierarchy

    def append(self, verts: XYVertices = None) -> None:
        if verts is not None:
            super().append(verts)

    def isEmpty(self):
        return len(self.stack()) == 0

    @property
    def x(self):
        return [lst.x for lst in self]

    @x.setter
    def x(self, newX):
        for lst, newLstX in zip(self, newX):
            lst.x = newLstX

    @property
    def y(self):
        return [lst.y for lst in self]

    @y.setter
    def y(self, newY):
        for lst, newLstY in zip(self, newY):
            lst.y = newLstY

    def asPoint(self):
        if len(self) == 1 and self[0].shape[0] == 1:
            return self[0][0]
        else:
            raise ValueError(
                f"Can only treat ComplexXYVertices with one inner list as a point."
                f" Current list has {len(self)} element(s), "
                f"where element 0 has shape {self[0].shape}."
            )

    def stack(self, newDtype=None) -> XYVertices:
        if len(self) == 0:
            # Check required for np vstack since it won't work with a 0-element array
            return XYVertices()
        else:
            if newDtype is None:
                newDtype = self[0].dtype
            return XYVertices(np.vstack(self), dtype=newDtype)

    def removeOffset(self, offset=None, inplace=False, returnOffset=False):
        """
        Subtracts a constant offset from all contained vertices. If offset is None,
        it will be set to the min value across all contained vertices such that this
        polygon's coordinates are all relative to (0,0).
        """
        if offset is None:
            offset = self.stack().min(0)
        if inplace:
            for verts in self:
                verts -= offset
            if returnOffset:
                return offset
        else:
            out = ComplexXYVertices([verts - offset for verts in self], self.hierarchy)
            if returnOffset:
                return out, offset
            return out

    def filledVerts(self) -> ComplexXYVertices:
        """
        Retrieves all vertex lists corresponding to filled regions in the complex shape
        """
        idxs = np.nonzero(self.hierarchy[:, 3] == -1)[0]
        return ComplexXYVertices([self[ii] for ii in idxs])

    def holeVerts(self) -> ComplexXYVertices:
        """
        Retrieves all vertex lists corresponding to holes in the complex shape
        """
        idxs = np.nonzero(self.hierarchy[:, 3] != -1)[0]
        return ComplexXYVertices([self[ii] for ii in idxs])

    def toMask(
        self,
        maskShape: Union[Sequence, NChanImg] = None,
        fillColor: Union[int, float, np.ndarray] = None,
        onlyFillConnected=False,
        warnIfTooSmall=True,
    ):
        if maskShape is None:
            try:
                maskShape = tuple(self.stack().max(0)[::-1] + 1)
            except ValueError:
                # Mask is zero-sized
                return np.zeros((0, 0), "uint16")
            # Guaranteed not to be too small
            warnIfTooSmall = False
        if warnIfTooSmall:
            cmpShape = (
                maskShape if isinstance(maskShape, Sequence) else maskShape.shape[:2]
            )
            # Wait until inside 'if' so max isn't unnecessarily calculated
            # Edge case: Empty vertices set will trigger a value warning
            if len(self) == 0:
                vertMax = 0
            else:
                vertMax = self.stack().max(0)[::-1]
            if np.any(vertMax > np.array(cmpShape[:2])):
                warn(
                    "Vertices don't fit in the provided mask size.\n"
                    f"Vertex shape: {vertMax}, mask shape: {cmpShape}",
                    stacklevel=2,
                )
        if onlyFillConnected:
            fillArg = []
            for verts in self:  # type: XYVertices
                if verts.connected:
                    fillArg.append(verts)
                else:
                    # Make sure each point is treated separately, not part of a shape
                    # to fill
                    fillArg.extend(verts)
        else:
            fillArg = self
        if isinstance(maskShape, NChanImg):
            out = maskShape
        else:
            out = np.zeros(maskShape, "uint16")
        nChans = 1 if out.ndim < 3 else out.shape[2]
        if fillColor is None:
            fillColor = tuple([1 for _ in range(nChans)])
        fillColorCmp = np.array(fillColor)
        if np.any(
            (np.iinfo(out.dtype).min > fillColorCmp)
            | (fillColorCmp > np.iinfo(out.dtype).max)
        ):
            raise ValueError(
                "Fill color is larger or smaller than mask range can represent"
            )
        cv.fillPoly(out, fillArg, fillColor)
        return out

    @staticmethod
    def fromBinaryMask(
        mask: BlackWhiteImg, approximation=cv.CHAIN_APPROX_SIMPLE, externalOnly=False
    ) -> ComplexXYVertices:
        """
        Creates ComplexXYVertices from a numpy binary image

        Parameters
        ----------
        mask
            Mask to locate connected components. Nonzero pixels are considered part
            of the component.
        approximation
            One of "CHAIN_APPROX_*" from opencv constants. ``None`` is an alias for
            CHAIN_APPROX_NONE.
        externalOnly
            If *True*, finds contours using RETR_EXTERNAL, otherwise uses RETR_CCOMP.
        """
        if approximation is None:
            approximation = cv.CHAIN_APPROX_NONE
        retrMethod = cv.RETR_CCOMP
        if externalOnly:
            retrMethod = cv.RETR_EXTERNAL
        # Contours are on the inside of components, so dilate first to make sure they
        # are on the outside
        # bwmask = dilation(bwmask, np.ones((3,3), dtype=bool))
        if mask.dtype != np.uint8:
            mask = mask.astype("uint8")
        contours, hierarchy = cv.findContours(mask, retrMethod, approximation)
        compVertices = ComplexXYVertices()
        for contour in contours:
            compVertices.append(XYVertices(contour[:, 0, :]))
        if hierarchy is None:
            hierarchy = np.ones((0, 1, 4), int) * -1
        else:
            hierarchy = hierarchy[0, :, :]
        compVertices.hierarchy = hierarchy
        return compVertices

    @staticmethod
    def _diagonalPairsToCorners(poly: T) -> T:
        """
        cv2.findContours has a habit of turning right-angles into pairs of 45-degree
        offset contour indices. This becomes a problem when using simplification methods
        like "approxPolyDP", since one of these edges is thrown out and the result is
        no longer rectangular. This method searches for pairs of 45-degree legs with
        side lengths of 1 to convert them into a single coordinate that makes proper
        right-angles. The resulting call to "fillPoly" produces an equivalent result,
        and "approxPolyDP" maintains rectangular shapes.

        poly
            Nx2 XYVertices to check (works with regular ndarray too)

        Returns
        -------
        Any
            Copy of ``poly`` with a single right-angle for every pair found as
            described in the documentation above.
        """
        polyDiff = np.diff(poly, axis=0, append=poly[[0]])
        # Cutoff corner segments are 45 degrees and sqrt(2) length long
        # So, find segments whose x and y differenceImages are both 1 and form 45-degree angles
        # 45 degree angles must also occur in pairs
        sqrt2Segs = np.all(np.abs(polyDiff) == 1, axis=1)
        segmentAngles = np.rad2deg(np.arctan2(polyDiff[:, 1], polyDiff[:, 0]))
        angleDiffs = np.diff(segmentAngles, prepend=segmentAngles[-1])
        # Ensure each angle is comparable using the least-magnitude representation
        # (Adding/subtracting 360 degrees doesn't change the angle)
        angleDiffsStackedTmp = angleDiffs[:, None] + np.array([[0, 360, -360]])
        angleDiffs = angleDiffsStackedTmp[
            np.arange(len(angleDiffs)), np.abs(angleDiffsStackedTmp).argmin(axis=1)
        ]
        diagAngles = np.bitwise_and(
            np.abs(angleDiffs) == 45, angleDiffs == np.roll(angleDiffs, -1)
        )

        replaceInds = np.flatnonzero(sqrt2Segs & diagAngles)
        # Also, curves should only be replaced where the diagonals share one coordinate
        # with a previous vertex (i.e. continues a line)
        # False positive
        # noinspection PyTypeChecker
        sharesPrevCoord: np.ndarray = poly[replaceInds] == poly[replaceInds - 1]
        # One shared and one non-shared coordinate
        falsePositives = np.sum(sharesPrevCoord, axis=1) != 1
        replaceInds = replaceInds[~falsePositives]
        # Negative angles curve inward, positive curve outward. If we want to always
        # prefer the outer edge, keep the first point in the pair during a positive
        # angle change and the second point for negative curve
        keepFirstPt = angleDiffs[replaceInds] > 0

        keepInds = np.where(keepFirstPt, replaceInds, replaceInds + 1)
        deleteInds = np.where(keepFirstPt, replaceInds + 1, replaceInds)
        # To replace a diagonal with a corner, substitute either the leading or
        # following point's coordinates depending on whether the diagonal curves inward
        # or outward "argmax" will point to either x or y coordinate depending on which
        # is *not* shared
        replaceDims = sharesPrevCoord[~falsePositives].argmin(axis=1)
        # If the following instead of leading edge is preferred, keep the opposite
        # coordinate since the rules for the point are swapped
        replaceDims[~keepFirstPt] = 1 - replaceDims[~keepFirstPt]

        returnPoly = poly.copy()
        returnPoly[keepInds, replaceDims] = poly[deleteInds, replaceDims]
        # Corners are turned into one vertex, so delete the far side and replce the near
        # side with the corner itself
        returnPoly = np.delete(returnPoly, deleteInds, axis=0)
        return returnPoly

    def simplify(self, epsilon=1.0) -> ComplexXYVertices:
        """
        Uses ``cv.approxPolyDP`` to reduce the number of vertices in each subregion.

        Parameters
        ----------
        epsilon
            Passed to ``cv.approxPolyDP``. Two special cases: if 0, diagonal pairs of
            vertices in each polygon are converted into corners. < 0 is considered a
            no-op.

        Returns
        -------
        ComplexXYVertices
            Simplified version of ComplexXYVertices
        """
        # Special case: No simplification performed
        if epsilon < 0:
            return self.copy()
        out = type(self)(hierarchy=self.hierarchy)
        for contour in self:  # type: XYVertices
            # See function documentation for justification of this step
            contour = self._diagonalPairsToCorners(contour)
            if epsilon > 0:
                contour = cv.approxPolyDP(contour, epsilon, contour.connected)[:, 0, :]
            out.append(XYVertices(contour))
        return out

    def __str__(self) -> str:
        """
        Improve the readability of vertex list in table by just displaying stats of
        larger arrays

        Returns
        -------
        str
            Human-readable string representation
        """
        concatVerts = self.stack()
        if len(concatVerts) <= 4:
            return str(concatVerts)
        return (
            f"Mean:\t{np.round(concatVerts.mean(0), 1)}\n"
            f"# Points:\t{len(concatVerts)}\n"
            f"Min:\t{concatVerts.min(0)}\n"
            f"Max:\t{concatVerts.max(0)}"
        )

    def __eq__(self, other: ComplexXYVertices):
        try:
            if len(self) != len(other):
                return False
            for selfVerts, otherVerts in zip(self, other):
                if not np.array_equal(selfVerts, otherVerts):
                    return False
        except TypeError:
            # No 'len()' or no 'iter()'
            return False
        return True

    def __ne__(self, other):
        return not self == other

    def copy(self) -> ComplexXYVertices:
        return ComplexXYVertices([lst.copy() for lst in self], self.hierarchy)

    def serialize(self):
        return str([arr.tolist() for arr in self])

    @staticmethod
    def deserialize(stringObject: str) -> ComplexXYVertices:
        # TODO: Infer appropriate hierarchy from the serialized string. It is possible
        #  by finding whether vertices are given in CW or CCW order. This doesn't
        #  affect how they are drawn, but it does effect the return values of
        #  "holeVerts()" and "filledVerts()"
        outerLst = literal_eval(stringObject)
        return ComplexXYVertices([XYVertices(lst) for lst in outerLst])


@pd.api.extensions.register_series_accessor("s3averts")
class S3AVertsAccessor:
    def __init__(self, verticesSeries: pd.Series):
        self.verts = verticesSeries

    def max(self):
        """
        Returns the max along dimension 0 for a list of complex vertices
        """
        return np.vstack([v.stack() for v in self.verts]).max(0)

    def min(self):
        """
        Returns the min along dimension 0 for a list of complex vertices
        """
        return np.vstack([v.stack() for v in self.verts]).min(0)

    def split(self):
        """
        Makes a separate component for each distinct boundary in all selected
        components. For instance, if two components are selected, and each has two
        separate circles as vertices, then 4 total rows will exist after this operation.
        """
        newVerts = []
        newIds = []
        for idx, verts in self.verts.items():
            tmpMask = verts.toMask().astype("uint8")
            nComps, ccompImg = cv.connectedComponents(tmpMask)
            for ii in range(1, nComps):
                newVerts.append(ComplexXYVertices.fromBinaryMask(ccompImg == ii))
                newIds.append(idx)
        return pd.Series(newVerts, index=newIds, name=self.verts.name)

    def merge(self):
        """
        Forms one single ComplexXYVertices object from a series of ComplexXYVertices
        regions. Overlapping is allowed. The output is determined by placing all
        vertices in a binary image and calling ``ComplexXYVertices.fromBinaryMask`` on
        the final mask.
        """
        # x-y -> row-col
        offset = self.min()
        maskShape = (self.max() - offset + 1)[::-1]
        mask = np.zeros(maskShape, "uint8")
        for verts in self.verts:  # type: ComplexXYVertices
            verts.removeOffset(offset).toMask(mask)
        return ComplexXYVertices.fromBinaryMask(mask).removeOffset(-offset)

    def removeOverlap(self):
        """
        Makes sure all specified vertices have no overlap. Preference is given
        in order of the given IDs, i.e. the last ID in the list is guaranteed to
        keep its full shape.
        """
        wholeMask = np.zeros(self.max()[::-1] + 1, dtype="uint16")
        for ii, verts in enumerate(self.verts, 1):
            verts.toMask(wholeMask, ii)
        outVerts = []
        for ii in range(1, len(self.verts) + 1):
            verts = ComplexXYVertices.fromBinaryMask(wholeMask == ii)
            outVerts.append(verts)
        return pd.Series(outVerts, index=self.verts.index, name=self.verts.name)
