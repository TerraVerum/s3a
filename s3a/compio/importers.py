from __future__ import annotations

import ast
import errno
import inspect
import itertools
import json
import os
import typing as t
from datetime import datetime
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
from qtextras.typeoverloads import FilePath
from skimage import draw

from .base import AnnotationImporter
from .helpers import registerIoHandler
from ..constants import REQD_TBL_FIELDS as RTF
from ..generalutils import DirectoryDict, cvImreadRgb, orderContourPoints, toDictGen
from ..structures import AnnInstanceError, ComplexXYVertices, LabelFieldType, XYVertices

__all__ = [
    "SerialImporter",
    "CsvImporter",
    "SuperannotateJsonImporter",
    "GeojsonImporter",
    "LblPngImporter",
    "PklImporter",
    "CompImgsDfImporter",
    "VGGImageAnnotatorImporter",
    "YoloV5Importer",
]


class SerialImporter(AnnotationImporter):
    ioType = "s3a"

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
        kwargs.update(na_filter=False, dtype=str)
        acceptedArgs = inspect.signature(importFn).parameters
        useArgs = kwargs.keys() & acceptedArgs
        serialDf = importFn(file, **{k: kwargs[k] for k in useArgs})
        return serialDf

    def bulkImport(self, importObj, **kwargs):
        return importObj

    def getInstances(self, importObj, **kwargs):
        return toDictGen(importObj)

    @staticmethod
    def _getPandasImporters():
        members = [v for v in vars(pd) if v.startswith("read_")]
        return [mem.replace("read_", "") for mem in members]


class CsvImporter(SerialImporter):
    # Override to provide custom access to csv file type
    pass


class GeojsonImporter(AnnotationImporter):
    def readFile(self, file: FilePath, **kwargs):
        with open(Path(file), "r") as ifile:
            return json.load(ifile)

    def getInstances(self, importObj, **kwargs):
        return importObj["features"]

    def formatSingleInstance(self, inst, **kwargs):
        return inst["properties"]

    @staticmethod
    def parseRegion(geometry):
        geometry = ast.literal_eval(geometry)
        if geometry["type"] == "Polygon":
            return ComplexXYVertices(geometry["coordinates"], coerceListElements=True)
        else:
            return AnnInstanceError(f'Unrecognized type "{geometry["type"]}"')

    bulkImport = AnnotationImporter.defaultBulkImport


registerIoHandler("geojsonregion", deserialize=GeojsonImporter.parseRegion)


class SuperannotateJsonImporter(AnnotationImporter):
    ioType = "superannotate"

    def readFile(self, file: FilePath, **kwargs):
        with open(Path(file), "r") as ifile:
            return json.load(ifile)

    def populateMetadata(
        self,
        file: Path = None,
        source: t.Union[FilePath, dict] = None,
        imageShape: tuple[int, int] = None,
        **kwargs,
    ):
        if source is None:
            source = file.parent
        source = DirectoryDict(source, readFunc=self.readFile, allowAbsolute=True)
        classes = source.get("classes.json")
        if classes is None and file is not None:
            classes = source.get(file.parent / "classes" / "classes.json")
        if classes is not None:
            self.tableData.fieldFromName("className").opts["limits"] = [
                c["name"] for c in classes
            ]
        meta = self.importObj["metadata"]
        if imageShape is None and "height" in meta and "width" in meta:
            imageShape = (meta["height"], meta["width"])
        return self._forwardMetadata(locals())

    def getInstances(self, importObj, **kwargs):
        return importObj["instances"]

    def bulkImport(self, importObj, source=None, **kwargs) -> pd.DataFrame:
        df = super().defaultBulkImport(importObj, **kwargs, source=source)
        df[RTF.IMAGE_FILE] = importObj["metadata"]["name"]
        return df

    def formatSingleInstance(self, inst, name=None, source=None, **kwargs):
        out = {}
        verts = self.parseRegion(inst)
        if not isinstance(verts, AnnInstanceError):
            verts = ComplexXYVertices([verts])
        # Need to serialize since wrapper function tries to deserialize
        out[RTF.VERTICES] = verts
        return out

    @staticmethod
    def parseRegion(inst: dict) -> t.Union[XYVertices, AnnInstanceError]:
        typ = inst["type"]
        if typ == "polygon":
            pts = inst["points"]
            pts = np.column_stack([pts[::2], pts[1::2]])
        elif typ in ("bbox", "rbbox"):
            dictPts = inst["points"]

            pts = []
            for number, plane in itertools.product(range(1, 5), ["x", "y"]):
                kk = f"{plane}{number}"
                if kk not in dictPts:
                    break
                pts.append(dictPts[kk])
            # x-y list is not formatted like a box -- fix this
            pts = np.column_stack([pts[::2], pts[1::2]])
            wh = np.diff(pts, axis=0)
            if len(pts) == 2:
                pts = wh * [[0, 0], [1, 0], [1, 1], [0, 1]] + pts[0]
        elif typ == "ellipse":
            vals = inst["cy"], inst["cx"], inst["ry"], inst["rx"], inst["angle"]
            pts = draw.ellipse_perimeter(*(int(v) for v in vals))
            pts = np.column_stack(pts[::-1])
            pts = orderContourPoints(pts)
        else:
            pts = AnnInstanceError(f'Unrecognized type "{typ}"')
        if not isinstance(pts, AnnInstanceError):
            pts = pts.view(XYVertices)
        return pts

    @staticmethod
    def parseAttributes(attrs):
        if attrs is None:
            attrs = []
        combined = "\n".join(a["groupName"] for a in attrs)
        return combined

    @staticmethod
    def parseTime(val):
        parsedTime = datetime.strptime(val, "%Y-%m-%dT%H:%M:%S.%fZ")
        return str(parsedTime)


registerIoHandler(
    "superannattributes", deserialize=SuperannotateJsonImporter.parseAttributes
)
registerIoHandler(
    "superanntime",
    deserialize=SuperannotateJsonImporter.parseTime,
)


class VGGImageAnnotatorImporter(CsvImporter):
    ioType = None  # Will be auto-assigned in init

    def formatSingleInstance(self, inst, **kwargs):
        out = json.loads(inst["region_attributes"])
        return out

    def getInstances(self, importObj: pd.DataFrame, **kwargs):
        return toDictGen(importObj)

    @staticmethod
    def parseRegion(region):
        region = json.loads(region)
        if not region:
            return ComplexXYVertices()
        name = region["name"]
        if name in ["polygon", "polyline"]:
            pts = XYVertices(
                np.column_stack((region["all_points_x"], region["all_points_y"]))
            )
            if name == "polyline":
                pts.connected = False
        elif name == "ellipse":
            vals = (
                region["cy"],
                region["cx"],
                region["ry"],
                region["rx"],
                region.get("theta", 0),
            )
            pts = draw.ellipse_perimeter(*(int(v) for v in vals[:-1]), vals[-1])
            pts = np.column_stack(pts[::-1])
            pts = orderContourPoints(pts)
        elif name == "rect":
            x, y = region["x"], region["y"]
            width, height = region["width"], region["height"]
            pts = XYVertices(
                [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]
            )
        elif name == "circle":
            cx, cy, r = region["cx"], region["cy"], region["r"]
            pts = draw.circle_perimeter(int(cy), int(cx), int(r))
            pts = np.column_stack(pts[::-1])
            pts = orderContourPoints(pts)
        elif name == "point":
            cx, cy = region["cx"], region["cy"]
            pts = XYVertices([[cx, cy]])
        else:
            raise ValueError(f'Unknown region shape: "{name}')
        return ComplexXYVertices([pts])


registerIoHandler("viaregion", deserialize=VGGImageAnnotatorImporter.parseRegion)


class LblPngImporter(AnnotationImporter):
    ioType = "s3a"

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
            into field values. See ``PrjParam.toNumeric`` for details, since this is the
            mapping expected. If not provided, first the image metadata tags are
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
        labelImage = self.importObj
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

    def getInstances(self, importObj, labelMap=None, distinctRegions=None, **kwargs):
        labelMask = importObj
        for numericLbl, origVal in labelMap.items():  # type: int, t.Any
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
    ioType = "s3a"

    def readFile(self, file: FilePath, **importArgs) -> pd.DataFrame:
        """
        See docstring for :func:`self.importCsv`
        """
        return pd.read_pickle(file)

    def getInstances(self, importObj, **kwargs):
        return toDictGen(importObj)

    def bulkImport(self, importObj, **kwargs):
        return importObj


class CompImgsDfImporter(AnnotationImporter):
    ioType = "s3a"

    readFile = PklImporter.readFile

    def getInstances(self, importObj, **kwargs):
        return importObj.iterrows()

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

    def bulkImport(self, importObj, labelField=None, **kwargs):
        out = importObj[["instanceId", "label"]].copy()
        out.columns = [RTF.ID, labelField]
        return out


class YoloV5Importer(CsvImporter):
    def readFile(self, file: FilePath, **kwargs):
        return pd.read_csv(
            file,
            sep=r"\s+",
            header=None,
            names=["class", "center_x", "center_y", "width", "height"],
        )

    def populateMetadata(self, imageShape=None, labelMap=None, **kwargs):
        if imageShape is None:
            raise ValueError("Must specify ``imageShape`` when importing yolov5 data")
        return self._forwardMetadata(locals())

    def bulkImport(self, importObj, imageShape=None, labelMap=None, **kwargs):
        imageShapeXy = np.array(imageShape[::-1])
        # Add 3rd dimension so all computations can be simultaneous
        # (N, 1, 2) wh * (1, 4, 2) boundingBox = (N,4,2) bboxes for N components
        # Thus, each row of ``bboxes`` is (4,2) boundingBox array
        centerXy = (
            importObj[["center_x", "center_y"]].to_numpy("float32")[:, None, :]
            * imageShapeXy
        )
        widthHeight = (
            importObj[["width", "height"]].to_numpy("float32")[:, None, :]
            * imageShapeXy
        )
        bboxes = (
            np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype="float32")[None, ...]
            * widthHeight
        )
        mins = centerXy - (widthHeight / 2)
        bboxes += mins
        # reshape to (N,) ComplexXyVertices
        verts = [ComplexXYVertices([XYVertices(box)]) for box in bboxes]
        out = pd.DataFrame()
        out[RTF.VERTICES] = verts
        classVals = importObj["class"].astype(int)
        if labelMap is not None:
            classVals = labelMap[classVals]
        out["class"] = classVals
        return out
