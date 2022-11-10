from __future__ import annotations

import errno
import inspect
import json
import os
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
from qtextras.typeoverloads import FilePath

from .base import AnnotationImporter
from ..constants import REQD_TBL_FIELDS as RTF
from ..generalutils import cvImreadRgb, toDictGen
from ..structures import ComplexXYVertices, LabelFieldType

__all__ = [
    "SerialImporter",
    "CsvImporter",
    "LblPngImporter",
    "PklImporter",
    "CompImgsDfImporter",
]


class SerialImporter(AnnotationImporter):
    @classmethod
    def readFile(cls, file: FilePath, fallbackFormat="csv", **kwargs):
        file = Path(file)
        if not file.exists():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(file))
        fType = file.suffix.lower().replace(".", "")
        if not fType and fallbackFormat:
            fType = fallbackFormat
        importFn = getattr(pd, f"read_{fType}", None)
        if importFn is None:
            raise ValueError(
                f"File type {fType} cannot be handled by the serial importer."
                f' Must be one of {",".join(cls._getPandasImporters())}'
            )
        # Special case: csv imports need to avoid interpreting nan results
        kwargs.setdefault("na_filter", False)
        kwargs.setdefault("dtype", str)
        acceptedArgs = inspect.signature(importFn).parameters
        useArgs = kwargs.keys() & acceptedArgs
        serialDf = importFn(file, **{k: kwargs[k] for k in useArgs})
        return serialDf

    def bulkImport(self, importObject, **kwargs):
        return importObject

    def getInstances(self, importObject, **kwargs):
        return toDictGen(importObject)

    @staticmethod
    def _getPandasImporters():
        members = [v for v in vars(pd) if v.startswith("read_")]
        return [mem.replace("read_", "") for mem in members]


class CsvImporter(SerialImporter):
    # Override to provide custom access to csv file type
    pass


class LblPngImporter(AnnotationImporter):
    imageInfo = {}

    def readFile(self, file: FilePath, labelMap=None, offset=0, **kwargs):
        try:
            image: Image.Image = Image.open(file)
            self.imageInfo = image.info
            # False positive
            # noinspection PyTypeChecker
            image = np.asarray(image)
        except TypeError:
            # E.g. float image
            return cvImreadRgb(str(file), mode=cv.IMREAD_UNCHANGED)
        return image

    def populateMetadata(
        self,
        labelField: LabelFieldType = "Instance ID",
        labelMap: pd.Series = None,
        distinctRegions=True,
        offset=0,
        **kwargs,
    ):
        """
        Parameters
        ----------
        labelField
            label field to associate with this image. Pixel values within the image
            correspond to values from this field in the table data. If *None*, this is
            inferred by the mapping read from the image file (see `labelMap`
            description)
        labelMap
            For parameters that aren't numeric and don't have limits (e.g. arbitrary
            string values), this mapping determines how numeric values should be turned
            into field values. See ``OptionsDict.toNumeric`` for details, since this is
            the mapping expected. If not provided, first the image metadata tags are
            searched for a 'labelMap' text attribute (this is often added to label
            images saved by S3A). Note that metadata can only be read from the file if
            a file path is provided, of course. If this check fails, it is inferred
            based on the allowed options of `labelField` (`labelField.opts['limits']`).
            Finally, if this is not present, it is assumed the raw image values can be
            used directly as field values.
        offset
            When ``labelMap`` is not provided and field values are directly inferred
            from label values, this determines whether (and how much if not *None*) to
            offset numeric labels during import. I.e. if the png label is 1, but offset
            is 1, the corresponding *field* value will be 0 (1 - offset = 0).
        distinctRegions
            Whether separate regions with the same ID should be separate IDs, or one ID
            with a group of polygons
        """
        # Rename for clarity
        labelImage = self.importObject
        # "Offset" present for numeric data, "mapping" present for textual data
        info = self.imageInfo
        if labelMap is None and "labelMap" in info:
            labelMap = pd.Series(
                json.loads(info["labelMap"]), name=info.get("labelField", None)
            )
            labelMap.index = labelMap.index.astype(int)

        if offset is None and "offset" in info:
            offset = int(info["offset"])

        labelField = self.tableData.fieldFromName(labelField or labelMap.name)
        if labelMap is None:
            vals = labelField.opts.get("limits", None) or np.unique(labelImage)
            _, labelMap = labelField.toNumeric(vals, returnMapping=True)
            labelMap.index += offset

        return self._forwardMetadata(locals())

    def getInstances(self, importObject, labelMap=None, distinctRegions=None, **kwargs):
        labelMask = importObject
        for numericLbl, origVal in labelMap.items():
            numericLbl: int
            verts = ComplexXYVertices.fromBinaryMask(labelMask == numericLbl)
            if distinctRegions:
                for vv in verts:
                    yield {
                        RTF.ID: numericLbl,
                        RTF.VERTICES: ComplexXYVertices([vv]),
                    }
            else:
                yield {RTF.ID: numericLbl, RTF.VERTICES: verts}

    bulkImport = AnnotationImporter.defaultBulkImport


class PklImporter(AnnotationImporter):
    def readFile(self, file: FilePath, **importArgs) -> pd.DataFrame:
        """
        See docstring for :func:`self.importCsv`
        """
        return pd.read_pickle(file)

    def getInstances(self, importObject, **kwargs):
        return toDictGen(importObject)

    def bulkImport(self, importObject, **kwargs):
        return importObject


class CompImgsDfImporter(AnnotationImporter):
    readFile = PklImporter.readFile

    def getInstances(self, importObject, **kwargs):
        return importObject.iterrows()

    def populateMetadata(self, labelField: LabelFieldType = "Instance ID", **kwargs):
        labelField = self.tableData.fieldFromName(labelField)
        ret = super().populateMetadata(**kwargs)
        ret.update(self._forwardMetadata(locals()))
        return ret

    def formatSingleInstance(self, inst, **kwargs) -> dict:
        idx, row = inst
        out = {}
        mask = row.labelMask
        verts = ComplexXYVertices.fromBinaryMask(mask).remove(-row.offset)
        out[RTF.VERTICES] = verts
        return out

    def bulkImport(self, importObject, labelField=None, **kwargs):
        out = importObject[["instanceId", "label"]].copy()
        out.columns = [RTF.ID, labelField]
        return out
