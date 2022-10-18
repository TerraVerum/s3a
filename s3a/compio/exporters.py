from __future__ import annotations

import errno
import inspect
import json
import os
import sys
import tempfile
import typing as t
from contextlib import ExitStack
from pathlib import Path
from stat import S_IREAD, S_IRGRP, S_IROTH
from zipfile import ZipFile

import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from utilitys import PrjParam
from utilitys import fns
from utilitys.typeoverloads import FilePath

from .base import AnnotationExporter, NO_ERRORS
from .helpers import serialize
from ..constants import REQD_TBL_FIELDS as RTF, PRJ_ENUMS
from ..generalutils import (
    subImageFromVerts,
    getCroppedImg,
    DirectoryDict,
    cvImreadRgb,
    toDictGen,
    cvImsaveRgb,
    imgPathtoHtml,
    toHtmlWithStyle,
)
from ..structures import ComplexXYVertices, PrjParamGroup

__all__ = [
    "LblPngExporter",
    "CompImgsDfExporter",
    "CompImgsZipExporter",
    "SerialExporter",
    "CsvExporter",
    "PklExporter",
    "YoloV5Exporter",
]


class LblPngExporter(AnnotationExporter):
    mapping: pd.Series = None
    inverseMapping: pd.Series = None

    def createExportObj(
        self, imageShape=None, backgroundColor=None, colormap=None, **_kwargs
    ):
        shape = imageShape[:2]
        if colormap:
            maskType = "uint8"
            shape = (*shape, 3)
        elif np.min(backgroundColor) >= 0:
            maskType = "uint16"
        else:
            maskType = "int32"
        return np.full(shape, backgroundColor, dtype=maskType)

    def populateMetadata(
        self,
        file: FilePath = None,
        imageShape: tuple[int] = None,
        labelField: PrjParam | str = "Instance ID",
        backgroundColor=0,
        offset: np.ndarray = None,
        colormap: str = None,
        returnLabelMapping=False,
        writeMeta=True,
        **kwargs,
    ):
        """
        Adds metadata to the label png file exporter

        Parameters
        ----------
        file
            File to save this object to, *None* will not write out the output.
        imageShape
            MxN shape of image containing these annotations
        labelField
            Data field to use as an index label. E.g. "Class" will use the 'class'
            column, but any other column can be specified. The output ground truth
            masks will be colored according to this field.  See
            :meth:`PrjParam.toNumeric` for details. If `labelField` is *None*,
            the foreground mask will be boolean instead of integer-colored.
        backgroundColor
            Color of the mask background. Must be an integer.
        offset
            For label png exports where an offset is added or subtracted, providing
            this information to the exporter allows that metadata to exist in the
            exported file
        colormap
            If provided, must correspond to a pyqtgraph colormap. Results
          in an RGB output instead of grayscale mask where each label is indexed into
          the colormap (including the background label). Note that providing a colormap
          prevents the export from being re-imported later by a LblPngImporter.
          type: str
          value: ''
        returnLabelMapping
            Whether to return a pd.Series matching original index values to their
            numeric counterparts. Note: this is important in cases where an offset must
            be applied to the underlying data. If the background color is 0 and a valid
            numeric value is also 0, it will be impossible to detect this object in the
            labeled output. So, an offset must be applied in these cases (background -
            min(data) + 1). This mapping records the relevant information to import
            original values back during `importLblPng`.
        writeMeta
            Whether to write the field mapping/offset to the output image file as png
            metadata. Useful to preserve label information when re-importing.
        **kwargs
            Additional keyword arguments to pass to
            :meth:`AnnotationExporter.populateMetadata`
        """
        labelField = PrjParamGroup.fieldFromParam(list(self.compDf.columns), labelField)
        if backgroundColor < 0 and not colormap:
            raise ValueError(f"Background color must be >= 0, was {backgroundColor}")

        if imageShape is None:
            # Without any components the image is non-existant
            if len(self.compDf) == 0:
                raise ValueError(
                    "imageShape cannot be *None* if no components are present"
                )
            vertMax = self.compDf[RTF.VERTICES].s3averts.max()
            imageShape = tuple(vertMax[::-1] + 1)

        return self._forwardMetadata(locals())

    def formatReturnObj(self, exportObj, returnLabelMapping=None, **kwargs):
        if returnLabelMapping:
            return exportObj, self.mapping
        return exportObj

    def writeFile(
        self,
        file: FilePath,
        exportObj,
        writeMeta=None,
        offset=None,
        **kwargs,
    ):
        mapping = self.mapping
        if writeMeta:
            outImg = Image.fromarray(exportObj)
            info = PngInfo()
            if mapping is not None:
                info.add_text("labelMapping", json.dumps(mapping.to_dict()))
                info.add_text("labelField", str(mapping.name))
            if offset is not None:
                info.add_text("offset", str(offset))
            outImg.save(file, pnginfo=info)
        else:
            cvImsaveRgb(file, exportObj)

    def updateExportObj(self, inst: dict, exportObj, **kwargs):
        verts: ComplexXYVertices = inst[RTF.VERTICES]
        verts.toMask(
            exportObj,
            inst[PRJ_ENUMS.FIELD_LABEL],
            checkForDisconnectedVerts=False,
        )
        return exportObj

    def individualExport(
        self,
        compDf: pd.DataFrame,
        exportObj,
        labelField=None,
        **kwargs,
    ):
        labels, mapping, backgroundColor = self.resolveMappings(
            compDf[labelField], labelField, **kwargs
        )
        exportObj[:] = backgroundColor
        compDf = compDf.copy()
        compDf[PRJ_ENUMS.FIELD_LABEL] = labels
        return super().individualExport(compDf, exportObj, **kwargs)

    def resolveMappings(
        self,
        labels: pd.Series = None,
        labelField=None,
        backgroundColor=None,
        colormap=None,
        storeMappings=True,
        **_kwargs,
    ):
        labels_numeric, mapping = labelField.toNumeric(labels, returnMapping=True)
        if colormap:
            mapping, backgroundColor = self.labelsToLutSer(
                mapping.to_numpy(), colormap, backgroundColor
            )
        else:
            # Make sure numeric labels aren't the same as background, otherwise they
            # will be forever lost
            diff = max(backgroundColor - np.min(labels_numeric, initial=0) + 1, 0)
            mapping.index += diff
            labels_numeric += diff

            # TODO: Determine appropriate course of action for float values
            #   For now, just ensure everything is int-like
            if not np.issubdtype(labels_numeric.dtype, np.integer):
                labels_numeric = labels_numeric.astype(int)
                mapping.index = mapping.index.astype(int)
        if storeMappings:
            self.mapping = mapping
            # Silence pandas warning
            dtype = None if len(mapping) else object
            self.inverseMapping = pd.Series(
                data=list(mapping.index), index=mapping, dtype=dtype
            )
        if colormap:
            labels_numeric = list(self.inverseMapping[labels])
        return labels_numeric, mapping, backgroundColor

    @staticmethod
    def labelsToLutSer(uniqueLabels, colormap, backgroundIndex=0):
        """
        Creates a LUT from mapping values and associates each numeric label with a
        LUT entry. Useful for e.g. painting an RGB output rather than making a
        grayscale mask

        Parameters
        ----------
        uniqueLabels
            array of all possible label values. The ordering of labels
            here determines the LUT index for each label. If a label in ``labels`` is not
            present in ``uniqueLabels``, it will be given the background index.
        colormap
            String name of colormap to use, e.g. 'viridis'. Raises
            ``ValueError`` if colormap name is not recognized among pyqtgraph options
        backgroundIndex
            LUT index to use for the background label, i.e. allows
            labels to be distinct from a unique background color. May be negative, i.e.
            -1, to refer to colors at the end of the LUT. May be *None* to have
            no reserved background label. Note that if backgroundIndex is *None*,
            labels not in ``uniqueLabels`` will be given 0 as an index.

        Returns
        -------
        tuple
            (indexes, lut) tuple where ``indexes`` are the locations within
            the lut for each initial label
        """
        numUniques = len(uniqueLabels)
        cmap = fns.getAnyPgColormap(colormap, forceExist=True)
        lut = cmap.getLookupTable(nPts=len(uniqueLabels) + 1)  # + 1 for bg label
        if backgroundIndex is not None:
            # Handle negative indexers
            backgroundIndex %= len(lut)
            lutIdxEnum = np.delete(np.arange(numUniques + 1), backgroundIndex)
        else:
            backgroundIndex = 0
            lutIdxEnum = np.arange(numUniques)
        return (
            # Dataframe *would* make more sense conceptually, but series allows
            # this return value to function like traditional label mappings
            pd.Series(data=uniqueLabels, index=lut[lutIdxEnum].T.tolist()),
            lut[backgroundIndex],
        )


class CompImgsDfExporter(AnnotationExporter):
    cropperFunc: t.Callable
    mappings: t.Union[pd.Series, t.Dict[str, pd.Series]]
    srcDir: DirectoryDict
    labelMaskDir: dict

    allOutputColumns = (
        "instanceId",
        "image",
        "labelMask",
        "label",
        "numericLabel",
        "offset",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exportLblPng = LblPngExporter()

    @fns.dynamicDocstring(cols=list(allOutputColumns))
    def populateMetadata(
        self,
        srcDir: FilePath | dict | DirectoryDict = None,
        labelField="Instance ID",
        labelMaskDir: FilePath | dict | DirectoryDict = None,
        includeCols=allOutputColumns,
        prioritizeById=False,
        returnLabelMapping=False,
        missingOk=False,
        resizeOpts=None,
        **kwargs,
    ):
        """
        Populates the metadata for this exporter. This method is called automatically
        during export.

        Parameters
        ----------
        srcDir
            Where images corresponding to this dataframe are kept. Source image
            filenames are interpreted relative to this directory if they are not
            absolute. Alternatively, can be a dict of name to np.ndarray image mappings
        includeCols
            Which columns to include in the export list
            pType: checklist
            limits: {cols}
            value: {cols}
            expanded: False
        labelField
            See ``ComponentIO.exportLblPng``. This label is provided in the output
            dataframe as well, if specified.
        labelMaskDir
            Similar to ``srcDir``, this is where label masks can be found. If not
            specified, a new label mask is generated for each image based on its
            components in the exported component table
        prioritizeById
            Since the label image export is only one channel (i.e. grayscale), problems
            arise when there is overlap between components. Which one should be on top?
            If `prioritizeById` is *True*, higher ids are always on top of lower ids.
            So, if ID 1 is a small component and ID 2 is a larger component completely
            surrounding ID 1, ID 1's export will just look like ID 2's export. If
            *Fals*, the current component is always on top in its label mask. In the
            case where more than 2 components overlap, the other components are ordered
            by ID. So, in the previous scenario ID 1 will still show up on top of ID 2
            in its own exported mask despite being a lower ID, but ID 2 will be fully
            visible in its own export.
        missingOk
            Whether a missing image is acceptable. When no source image is found for an
            annotation, this will simpy the 'image' output property
        resizeOpts
            Options for reshaping the output to a uniform size if desired. The
            following keys may be supplied:
              - ``shape`` : Required. It is the shape that all images will be resized
                to before being saved. This is useful for neural networks with a fixed
                input size which forces all inputs to be e.g. 100x100 pixels.
              - ``keepAspectRatio``: default True. Whether to keep the aspect ratio and
                pad the problematic axis, or to stretch the image to the right fit. I.e.
                if a component with shape (25, 50) exists, and an export ``shape`` of
                (25, 25) is specified with ``keepAspectRatio``, the component will be
                resized to (12, 25) and padded on the top and bottom with 6 and 7 pixels
                of ``padVal``, respectively.
              - ``padVal``: default np.nan. How to fill the padded axis if
                `keepAspectRatio` is *True*. If *np.nan*, the values are grabbed from the
                image instead. If a component is on the image boundary, black (0) is
                used.
              - ``allowReorient``: default False. If *True*, the output image can be
                rotated 90 degrees if this reduces the amount of manipulation required
                to get the output to be the proper shape
              - ``interpolation``: Any interpolation value accepted by ``cv.resize``
        returnLabelMapping
            Whether to return the mapping of label numeric values to table field values

        Returns
        -------
        pd.DataFrame
            Dataframe with the following keys:
              - instId: The component's Instance ID
              - img: The (MxNxC) image corresponding to the component vertices, where
                MxN are the padded row sizes and C is the number of image channels
              - labelMask: Binary mask representing the component vertices
              - label: Field value of the component for the field specified by
                `labelField`
              - offset: Image (x,y) coordinate of the min component vertex.
        """
        if srcDir is None:
            srcDir = Path()
        imageReaderFunc = lambda file: cvImreadRgb(file, cv.IMREAD_UNCHANGED)
        srcDir = DirectoryDict(
            srcDir,
            allowAbsolute=True,
            readFunc=imageReaderFunc,
            cacheOnRead=False,
        )
        self.srcDir = srcDir
        # Label masks are programmatically generated so no need for a backing directory
        self.labelMaskDir = DirectoryDict(
            labelMaskDir,
            allowAbsolute=True,
            readFunc=imageReaderFunc,
            cacheOnRead=False,
        )
        if resizeOpts is not None:
            cropperFunc = subImageFromVerts
        else:
            resizeOpts = {}
            cropperFunc = getCroppedImg
        self.cropperFunc = cropperFunc
        labelField = PrjParamGroup.fieldFromParam(self.compDf, labelField)
        return self._forwardMetadata(locals())

    def writeFile(self, file: FilePath, exportObj, **kwargs):
        return exportObj.to_pickle(file)

    def formatReturnObj(self, exportObj, returnLabelMapping=False, **_kwargs):
        if returnLabelMapping:
            return exportObj, self.mappings
        return exportObj

    def createExportObj(self, **kwargs):
        return []

    def bulkExport(self, compDf, exportObj, **kwargs):
        # imageShape is automatically inferred by the exporter
        kwargs.pop("imageShape", None)
        # File is taken care of in outer scope
        kwargs.pop("file", None)
        mappings = {}

        exportIndexes = []
        for fullImgName, miniDf in compDf.groupby(
            RTF.IMG_FILE
        ):  # type: str, pd.DataFrame
            exportedComps, mapping = self._formatSingleImage(
                miniDf, fullImgName, **kwargs
            )
            mappings[Path(fullImgName).name] = mapping
            exportObj.extend(exportedComps)
            exportIndexes.extend(miniDf.index)

        exportObj = pd.DataFrame(exportObj, index=exportIndexes)
        if len(mappings) == 1:
            # Common case where annotations for just one image were converted
            mappings = next(iter(mappings.values()))
        exportObj.attrs["mapping"] = mappings
        self.mappings = mappings
        return exportObj, NO_ERRORS.copy()

    def _formatSingleImage(
        self,
        compDf: pd.DataFrame,
        imageName,
        labelMask=None,
        labelMapping=None,
        missingOk=False,
        **kwargs,
    ):
        exportObj = []
        img = self.srcDir.get(imageName)
        if img is None and not missingOk:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), imageName)
        shape = img if img is None else img.shape[:2]
        # Make sure no options are duplicated
        kwargs.update(imageShape=shape)
        labelMask, labelMapping = self._resolveLabelMaskAndMapping(
            labelMask, labelMapping, compDf, imageName, **kwargs
        )
        if img is None:
            img = np.zeros_like(labelMask)
        labelValues = self.exportLblPng.inverseMapping[compDf[kwargs["labelField"]]]
        for row, labelValue in zip(toDictGen(compDf), labelValues):
            exportObj.append(
                self._formatSingleComp(
                    row, image=img, labelMask=labelMask, labelValue=labelValue, **kwargs
                )
            )
        return exportObj, labelMapping

    def _resolveLabelMaskAndMapping(
        self, labelMask, labelMapping, compDf, imageName, **lblPngKwargs
    ):
        """
        Allows for any combination of missing mask, label mask, or both. Requires many
        edge cases, since masks can come from user, directory object, or be generated
        programmatically.
        """
        lblPngKwargs.update(returnLabelMapping=True)
        if labelMask is None:
            labelMask = self.labelMaskDir.get(imageName)
        # Label mask can still be none if there is no source directory hit
        if labelMask is None:
            return self.exportLblPng(compDf, **lblPngKwargs)
        # Else, Need to ensure lblPng mappings are up to date for numeric label retrieval
        if labelMapping is not None and "colormap" not in lblPngKwargs:
            self.exportLblPng.mapping = labelMapping
            self.exportLblPng.inverseMapping = pd.Series(
                data=labelMapping.index, index=labelMapping.to_numpy()
            )
        else:
            labelField = lblPngKwargs["labelField"]
            _, labelMapping, _ = self.exportLblPng.resolveMappings(
                compDf[labelField], labelField, **lblPngKwargs
            )

        return labelMask, labelMapping

    def _formatSingleComp(
        self,
        inst: t.Any,
        *,
        image=None,
        labelMask=None,
        labelField=None,
        labelValue=None,
        includeCols=None,
        prioritizeById=None,
        resizeOpts=None,
        returnStats=None,
        **_kwargs,
    ):
        out = {}
        allVerts = inst[RTF.VERTICES].stack()
        imageName = inst[RTF.IMG_FILE]
        if image is None:
            image = self.srcDir.get(imageName)
        if labelMask is None:
            labelMask = self.labelMaskDir.get(imageName)
        resizeOpts = resizeOpts.copy()
        returnStats = returnStats or resizeOpts.pop("returnStats", None)
        if returnStats:
            compImg, bounds, stats = self.cropperFunc(
                image,
                allVerts,
                returnCoords=True,
                returnStats=returnStats,
                **resizeOpts,
            )
        else:
            compImg, bounds = self.cropperFunc(
                image, allVerts, returnCoords=True, **resizeOpts
            )
            stats = None
        useKeys = includeCols

        if "instanceId" in useKeys:
            out["instanceId"] = inst[RTF.INST_ID]

        if "offset" in useKeys:
            out["offset"] = bounds[0, :]

        lbl = inst[labelField]
        if "label" in useKeys:
            out["label"] = lbl

        if "numericLabel" in useKeys:
            out["numericLabel"] = labelValue

        if stats is not None:
            out.update(stats)

        if "image" in useKeys:
            out["image"] = compImg

        if "labelMask" in useKeys:
            if prioritizeById:
                # ID indicates z-value, which is already the case for a label image
                useImg = labelMask
            else:
                # The current component should always be drawn on top
                useImg = inst[RTF.VERTICES].toMask(labelMask.copy(), labelValue)
            mask = self.cropperFunc(useImg, allVerts, returnCoords=False, **resizeOpts)

            out["labelMask"] = mask

        return out


class CompImgsZipExporter(CompImgsDfExporter):
    def populateMetadata(
        self, archive=False, makeSummary=False, summaryImageWidth=None, **kwargs
    ):
        """
        Parameters
        ----------
        archive
            Whether to compress into a zip archive instead of directly outputting a folder
        makeSummary
            Whether to include an html table showing each component from the dataframe
            along with its image and mask representations
        summaryImageWidth
            If `makeSummary` is *True*, this parameter can determine whether each image
            should have a fixed width in the summary table. This can be useful to
            normalize the display of all images. Should be in pixels
        """
        ret = super().populateMetadata(**kwargs)
        ret.update(self._forwardMetadata(locals()))
        return ret

    def writeFile(
        self,
        file: FilePath,
        exportObj,
        archive=None,
        makeSummary=None,
        summaryImageWidth=None,
        **kwargs,
    ):
        outDir = Path(file)
        useDir = outDir

        with ExitStack() as stack:
            if archive:
                useDir = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            dataDir = useDir / "data"
            labelsDir = useDir / "labels"
            dataDir.mkdir(exist_ok=True, parents=True)
            labelsDir.mkdir(exist_ok=True, parents=True)

            summaryName = useDir / "summary.html"

            for idx, row in exportObj.iterrows():
                saveName = f'{row["instanceId"]}.png'
                if "image" in row.index:
                    cvImsaveRgb(dataDir / saveName, row["image"])
                if "labelMask" in row.index:
                    cvImsaveRgb(labelsDir / saveName, row.labelMask)

            if makeSummary:
                self._createSummary(
                    exportObj,
                    useDir,
                    dataDir,
                    labelsDir,
                    summaryName,
                    summaryImageWidth,
                )
            else:
                summaryName = None

            if archive:
                self._createArchive(outDir, dataDir, labelsDir, summaryName)

    @staticmethod
    def _createArchive(parentDir, dataDir, labelsDir, summaryName):
        makeSummary = summaryName is not None
        if parentDir.suffix != ".zip":
            parentDir = parentDir.with_suffix(parentDir.suffix + ".zip")
        with ZipFile(parentDir, "w") as ozip:
            for dir_ in labelsDir, dataDir:
                if not dir_.exists():
                    continue
                for file in dir_.iterdir():
                    ozip.write(file, f"{dir_.name}/{file.name}")
            if makeSummary:
                ozip.write(summaryName, file.name)

    def _createSummary(
        self,
        exportObj,
        parentDir,
        dataDir,
        labelsDir,
        summaryName,
        imageWidth,
    ):
        extractedImgs = exportObj.rename({"instanceId": RTF.INST_ID.name}, axis=1)
        # Prevent merge error by renaming index
        # INST_ID.name has to be used instead of raw INST_ID due to strange pandas issue
        # throwing a TypeError: keywords must be a string
        outDf: pd.DataFrame = self.compDf.drop([RTF.VERTICES], axis=1).rename(
            str, axis=1
        )
        # Unset index name in case it clashes with Instance ID column
        outDf.index.name = None
        outDf = outDf.merge(extractedImgs, on=RTF.INST_ID.name)

        def imgFormatter(el):
            return imgPathtoHtml((relDir / str(el)).with_suffix(".png").as_posix())

        for colName, imgDir in zip(["labelMask", "image"], [labelsDir, dataDir]):
            if colName not in extractedImgs:
                continue
            relDir = imgDir.relative_to(parentDir)
            outDf[colName] = outDf[RTF.INST_ID.name].apply(imgFormatter)
        outDf.columns = list(map(str, outDf.columns))
        style = None
        if imageWidth is not None:
            style = inspect.cleandoc(
                f"""
                img {{
                  width: {imageWidth}px
                }}
                """
            )
        toHtmlWithStyle(outDf, summaryName, style, escape=False, index=False)


class SerialExporter(AnnotationExporter):
    """
    The file suffix can be any option supported by a ``pandas`` exporter. This can be csv,
    json, feather, etc. Note: pickle is a special case. In some cases,
    it is significantly more benficial to export the raw dataframe compared to a
    serialized version. In these cases, use ComponentIO.exportPkl. Otherwise, `pickle`
    is still a valid option here for a serialized format. For a full list of export
    options, see
    `the documentation`https://pandas.pydata.org/pandas-docs/stable /user_guide/io.html`.
    """

    def createExportObj(self, **kwargs):
        return self.compDf.copy()

    def populateMetadata(self, readonly=False, **kwargs):
        """
        Parameters
        ----------
        readonly
            Whether this export should be read-only
        """
        return self._forwardMetadata(locals())

    @classmethod
    def writeFile(cls, file: FilePath, exportObj, readonly=None, **kwargs):
        defaultExportParams = {
            "na_rep": "NaN",
            "float_format": "{:0.10n}",
            "index": False,
        }
        outPath = Path(file)
        exporter = outPath.suffix.lower().replace(".", "")

        defaultExportParams.update(kwargs)
        exportFn = getattr(exportObj, f"to_{exporter}", None)
        if exportFn is None:
            raise ValueError(
                f'Exporter "{exporter}" not recognized. Acceptable options:\n'
                + ", ".join(cls._getPdExporters())
            )

        with np.printoptions(threshold=sys.maxsize):
            exportFn(file, index=False)
        if readonly:
            outPath.chmod(S_IREAD | S_IRGRP | S_IROTH)

    @staticmethod
    def _getPdExporters():
        members = inspect.getmembers(
            pd.DataFrame,
            lambda meth: inspect.isfunction(meth) and meth.__name__.startswith("to_"),
        )
        return [mem[0].replace("to_", "") for mem in members]

    def bulkExport(self, compDf, exportObj, readonly=None, **kwargs):
        allErrs = []
        # TODO: Currently the additional options are causing errors. Find out why and fix
        #  them, since this may be useful if it can be modified
        # Format special columns appropriately
        # Since CSV export significantly modifies the df, make a copy before doing all
        # these operations
        for col in exportObj:
            if not isinstance(col, PrjParam):
                exportObj[col] = exportObj[col].apply(str)
            elif not isinstance(col.value, str):
                serial, errs = serialize(col, exportObj[col])
                exportObj[col] = serial.to_numpy()
                allErrs.append(errs)
        # Pandas raises error concatenating empty list
        if len(allErrs):
            allErrs = pd.concat(allErrs, axis=1)
        else:
            allErrs = NO_ERRORS.copy()
        return exportObj, allErrs


class CsvExporter(SerialExporter):
    pass


class PklExporter(AnnotationExporter):
    def writeFile(self, file: FilePath, exportObj, **kwargs):
        exportObj.to_pickle(file)

    def createExportObj(self, **kwargs):
        return self.compDf.copy()


class YoloV5Exporter(CsvExporter):
    ioType = "yolov5"
    mapping: pd.Series

    def writeFile(self, file: FilePath, exportObj: pd.DataFrame, **kwargs):
        kwargs.setdefault("float_format", "%.6f")

        classLen = exportObj["class"].astype(str).str.len().max()
        kwargs.setdefault("formatters", {"class": f"{{:<{classLen}}}".format})

        exportObj.to_string(file, index=False, header=False)

    def populateMetadata(
        self,
        labelField=None,
        imageShape=None,
        returnLabelMapping=False,
        **kwargs,
    ):
        if imageShape is None:
            raise ValueError("Cannot export yolov5 without specifying `imageShape`")
        if labelField is None:
            raise ValueError("Cannot export yolov5 without specifying `labelField`")
        labelField = PrjParamGroup.fieldFromParam(self.compDf, labelField)
        return self._forwardMetadata(locals())

    def bulkExport(
        self,
        compDf,
        exportObj,
        readonly=None,
        imageShape=None,
        labelField=None,
        **kwargs,
    ):
        numericVals, mapping = labelField.toNumeric(
            exportObj[labelField], returnMapping=True
        )
        self.mapping = mapping
        if not np.issubdtype(numericVals.dtype, np.integer):
            raise ValueError("Yolo export only supports integer class values")
        stacked = exportObj[RTF.VERTICES].apply(ComplexXYVertices.stack)
        mins = np.vstack(stacked.apply(lambda el: el.min(0)))
        ptps = np.vstack(stacked.apply(lambda el: el.ptp(0)))
        imageShapeXy = np.array(imageShape[::-1])[None, :]
        exportObj = pd.DataFrame()
        exportObj["class"] = numericVals
        exportObj[["center_x", "center_y"]] = (mins + (ptps / 2)) / imageShapeXy
        exportObj[["width", "height"]] = ptps / imageShapeXy
        return super().bulkExport(compDf, exportObj, readonly, **kwargs)

    def formatReturnObj(self, exportObj, returnLabelMapping=None, **kwargs):
        if returnLabelMapping:
            return exportObj, self.mapping
        return exportObj
