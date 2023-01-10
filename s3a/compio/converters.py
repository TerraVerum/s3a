from __future__ import annotations

import itertools
import json
import typing as t
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from qtextras.typeoverloads import FilePath
from skimage import draw

from .base import AnnotationImporter
from .exporters import CsvExporter
from .importers import CsvImporter
from ..constants import REQD_TBL_FIELDS as RTF
from ..generalutils import DirectoryDict, orderContourPoints, toDictGen
from ..structures import (
    AnnInstanceError,
    ComplexXYVertices,
    OptionsDictGroup,
    XYVertices,
)


class YoloV5Exporter(CsvExporter):
    ioTemplate = "yolov5"
    mapping: pd.Series

    def writeFile(self, file: FilePath, exportObject: pd.DataFrame, **kwargs):
        kwargs.setdefault("float_format", "%.6f")

        classLen = exportObject["class"].astype(str).str.len().max()
        kwargs.setdefault("formatters", {"class": f"{{:<{classLen}}}".format})

        exportObject.to_string(file, index=False, header=False)

    def populateMetadata(
        self,
        labelField=None,
        imageShape=None,
        returnLabelMap=False,
        **kwargs,
    ):
        if imageShape is None:
            raise ValueError("Cannot export yolov5 without specifying `imageShape`")
        if labelField is None:
            raise ValueError("Cannot export yolov5 without specifying `labelField`")
        labelField = OptionsDictGroup.fieldFromParameter(self.componentDf, labelField)
        return self._forwardMetadata(locals())

    def bulkExport(
        self,
        componentDf,
        exportObject,
        readonly=None,
        imageShape=None,
        labelField=None,
        **kwargs,
    ):
        numericVals, mapping = labelField.toNumeric(
            exportObject[labelField], returnMapping=True
        )
        self.mapping = mapping
        if not np.issubdtype(numericVals.dtype, np.integer):
            raise ValueError("Yolo export only supports integer class values")
        stacked = exportObject[RTF.VERTICES].apply(ComplexXYVertices.stack)
        mins = np.vstack(stacked.apply(lambda el: el.min(0)))
        ptps = np.vstack(stacked.apply(lambda el: el.ptp(0)))
        imageShapeXy = np.array(imageShape[::-1])[None, :]
        exportObject = pd.DataFrame()
        exportObject["class"] = numericVals
        exportObject[["center_x", "center_y"]] = (mins + (ptps / 2)) / imageShapeXy
        exportObject[["width", "height"]] = ptps / imageShapeXy
        return super().bulkExport(componentDf, exportObject, readonly, **kwargs)

    def formatReturnObject(self, exportObject, returnLabelMap=None, **kwargs):
        if returnLabelMap:
            return exportObject, self.mapping
        return exportObject


class GeojsonImporter(AnnotationImporter):
    ioTemplate = "geojson"

    def readFile(self, file: FilePath, **kwargs):
        with open(Path(file), "r") as ifile:
            return json.load(ifile)

    def getInstances(self, importObject, **kwargs):
        return importObject["features"]

    def formatSingleInstance(self, inst, **kwargs):
        out = dict(inst["properties"])
        out[RTF.VERTICES] = self.parseRegion(inst["geometry"])
        return out

    @staticmethod
    def parseRegion(geometry):
        if geometry["type"] == "Polygon":
            return ComplexXYVertices(geometry["coordinates"], coerceListElements=True)
        else:
            return AnnInstanceError(f'Unrecognized type "{geometry["type"]}"')

    bulkImport = AnnotationImporter.defaultBulkImport


class SuperannotateImporter(AnnotationImporter):
    ioTemplate = "superannotate"

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
        source = DirectoryDict(source, readFunction=self.readFile, allowAbsolute=True)
        classes = source.get("classes.json")
        if classes is None and file is not None:
            classes = source.get(file.parent / "classes" / "classes.json")
        if classes is not None:
            self.tableData.fieldFromName("className").opts["limits"] = [
                c["name"] for c in classes
            ]
        meta = self.importObject["metadata"]
        if imageShape is None and "height" in meta and "width" in meta:
            imageShape = (meta["height"], meta["width"])
        return self._forwardMetadata(locals())

    def getInstances(self, importObject, **kwargs):
        return importObject["instances"]

    def bulkImport(self, importObject, source=None, **kwargs) -> pd.DataFrame:
        df = super().defaultBulkImport(importObject, **kwargs, source=source)
        df[RTF.IMAGE_FILE] = importObject["metadata"]["name"]
        return df

    def formatSingleInstance(self, inst, name=None, source=None, **kwargs):
        out = {}
        verts = self.parseRegion(inst)
        if not isinstance(verts, AnnInstanceError):
            verts = ComplexXYVertices([verts])
        # Need to serialize since wrapper function tries to deserialize
        out[RTF.VERTICES] = verts
        if "createdAt" in inst:
            out["createdAt"] = self.parseTime(inst["createdAt"])
        if "attributes" in inst:
            out["attributes"] = self.parseAttributes(inst["attributes"])
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


class VGGImageAnnotatorImporter(CsvImporter):
    ioTemplate = "vggimageannotator"  # Will be auto-assigned in init

    def bulkImport(self, importObject, **kwargs):
        df = super().bulkImport(importObject, **kwargs)
        return df.drop(
            columns=["file_attributes", "region_shape_attributes", "region_attributes"]
        )

    def formatSingleInstance(self, inst, **kwargs):
        out = json.loads(inst["region_attributes"])
        out[RTF.VERTICES] = self.parseRegion(inst["region_shape_attributes"])
        return out

    def getInstances(self, importObject: pd.DataFrame, **kwargs):
        return toDictGen(importObject)

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

    def bulkImport(self, importObject, imageShape=None, labelMap=None, **kwargs):
        imageShapeXy = np.array(imageShape[::-1])
        # Add 3rd dimension so all computations can be simultaneous
        # (N, 1, 2) wh * (1, 4, 2) boundingBox = (N,4,2) bboxes for N components
        # Thus, each row of ``bboxes`` is (4,2) boundingBox array
        centerXy = (
            importObject[["center_x", "center_y"]].to_numpy("float32")[:, None, :]
            * imageShapeXy
        )
        widthHeight = (
            importObject[["width", "height"]].to_numpy("float32")[:, None, :]
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
        classVals = importObject["class"].astype(int)
        if labelMap is not None:
            classVals = labelMap[classVals]
        out["class"] = classVals
        return out
