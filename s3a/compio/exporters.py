from __future__ import annotations

import errno
import inspect
import json
import os
import sys
import tempfile
import textwrap
import typing as t
from contextlib import ExitStack
from pathlib import Path
from stat import S_IREAD, S_IRGRP, S_IROTH
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from skimage.exposure import rescale_intensity

from utilitys import PrjParam
from utilitys.fns import dynamicDocstring
from utilitys.typeoverloads import FilePath
from .base import AnnotationExporter, NO_ERRORS
from .helpers import serialize
from ..constants import REQD_TBL_FIELDS as RTF, PRJ_ENUMS
from ..structures import ComplexXYVertices
from ..generalutils import (
  subImageFromVerts,
  getCroppedImg,
  DirectoryDict,
  cvImread_rgb,
  pd_iterdict,
  cvImsave_rgb,
  imgPathtoHtml, pd_toHtmlWithStyle
)
from ..structures import PrjParamGroup

__all__ = ['LblPngExporter', 'CompImgsDfExporter', 'CompImgsZipExporter', 'SerialExporter', 'CsvExporter', 'PklExporter']

class LblPngExporter(AnnotationExporter):
  compDf: pd.DataFrame
  mapping: pd.Series

  def createExportObj(self,
                      imageShape=None,
                      backgroundColor=None,
                      **_kwargs
                      ):
    maskType = 'uint16' if np.min(backgroundColor) >= 0 else 'int32'
    return np.full(imageShape[:2], backgroundColor, dtype=maskType)

  def populateMetadata(
      self,
      file: FilePath=None,
      imageShape: tuple[int] = None,
      labelField: PrjParam | str = 'Instance ID',
      backgroundColor=0,
      offset: np.ndarray=None,
      rescaleOutput=False,
      returnLabelMapping=False,
      writeMeta=True,
      **kwargs
  ):
    """
    :param file: File to save this object to, *None* will not write out the output.
    :param imageShape: MxN shape of image containing these annotations
    :param labelField: Data field to use as an index label. E.g. "Class" will use the 'class'
      column, but any other column can be specified. The output ground truth masks
      will be colored according to this field.  See :meth:`PrjParam.toNumeric` for details.
      If `labelField` is *None*, the foreground mask will be boolean instead of integer-colored.
    :param backgroundColor: Color of the mask background. Must be an integer.
    :param offset: For label png exports where an offset is added or subtracted, providing this information
      to the exporter allows that metadata to exist in the exported file
    :param rescaleOutput: For images designed for human use, it is helpful to have
      outputs rescaled to the entire intensity range. Otherwise, they usually end
      up looking pure black and it is difficult to see components in the image.
      When `rescaleOutput` is *True*, all numbers are scaled to the 'uint16' range.
    :param returnLabelMapping: Whether to return a pd.Series matching original index values
      to their numeric counterparts. Note: this is important in cases where an offset must be applied to the underlying
      data. If the background color is 0 and a valid numeric value is also 0, it will be impossible to detect this
      object in the labeled output. So, an offset must be applied in these cases (background - min(data) + 1). This
      mapping records the relevant information to import original values back during `importLblPng`.
    :param writeMeta: Whether to write the field mapping/offset to the output image file as png metadata.
      Useful to preserve label information when re-importing.
    :param kwargs:
    :return:
    """
    labelField = PrjParamGroup.fieldFromParam(list(self.compDf.columns), labelField)
    compDf = self.compDf
    self.compDf = compDf.copy()

    if backgroundColor < 0:
      raise ValueError(f'Background color must be >= 0, was {backgroundColor}')

    readMapping = returnLabelMapping or (writeMeta and file is not None)
    if imageShape is None:
      # Without any components the image is non-existant
      if len(compDf) == 0:
        raise ValueError('imageShape cannot be *None* if no components are present')
      vertMax = ComplexXYVertices.stackedMax(compDf[RTF.VERTICES])
      imageShape = tuple(vertMax[::-1] + 1)

    return self._forwardMetadata(locals(), readMapping=readMapping)

  def formatReturnObj(self,
                      exportObj,
                      returnLabelMapping=None,
                      **kwargs):
    if returnLabelMapping:
      return exportObj, self.mapping
    return exportObj

  def writeFile(self,
                file: FilePath,
                exportObj,
                writeMeta=None,
                offset=None,
                **kwargs):
    mapping = self.mapping
    if writeMeta:
      outImg = Image.fromarray(exportObj)
      info = PngInfo()
      if mapping is not None:
        info.add_text('labelMapping', json.dumps(mapping.to_dict()))
        info.add_text('labelField', str(mapping.name))
      if offset is not None:
        info.add_text('offset', str(offset))
      outImg.save(file, pnginfo=info)
    else:
      cvImsave_rgb(file, exportObj)

  def updateExportObj(self, inst: dict, exportObj, **kwargs):
    verts: ComplexXYVertices = inst[RTF.VERTICES]
    verts.toMask(exportObj,
                 int(inst[PRJ_ENUMS.FIELD_LABEL]),
                 asBool=False,
                 checkForDisconnectedVerts=False)
    return exportObj

  def individualExport(self, compDf: pd.DataFrame, exportObj, **kwargs):
    labels_numeric, self.mapping = self._resolveMapping(**kwargs)
    self.compDf[PRJ_ENUMS.FIELD_LABEL] = labels_numeric
    asBool = np.issubdtype(labels_numeric.dtype, np.bool_)
    exportObj, errs = super().individualExport(self.compDf, exportObj, **kwargs)
    if asBool:
      exportObj = exportObj > 0
    return exportObj, errs

  def _resolveMapping(
      self,
      labelField=None,
      readMapping=None,
      rescaleOutput=None,
      backgroundColor=None,
      **_kwargs
  ):
    compDf = self.compDf
    labels = compDf[labelField]
    labels_numeric = labelField.toNumeric(labels, returnMapping=readMapping)
    # Make sure numeric labels aren't the same as background, otherwise they will be forever lost
    mapping = None
    if readMapping:
      labels_numeric, mapping = labels_numeric
    diff = max(backgroundColor - np.min(labels_numeric, initial=0) + 1, 0)
    if readMapping:
      mapping.index += diff
    labels_numeric += diff

    if rescaleOutput:
      if mapping is not None:
        max_ = np.max(np.asarray(mapping.index), initial=backgroundColor)
        mapping.index = rescale_intensity(
          mapping.index, in_range=(backgroundColor, max_), out_range='uint16'
        )
      else:
        max_ = np.max(labels_numeric, initial=backgroundColor)

      # False positive
      # noinspection PyTypeChecker
      labels_numeric: np.ndarray = rescale_intensity(
        labels_numeric, in_range=(backgroundColor, max_), out_range='uint16'
      )
    return labels_numeric, mapping

class CompImgsDfExporter(AnnotationExporter):
  cropperFunc: t.Callable
  invertedMap: pd.Series
  mappings: t.Union[pd.Series, t.Dict[str, pd.Series]]
  srcDir: DirectoryDict
  labelMaskDir: dict

  allOutputColumns = ('instanceId', 'image', 'labelMask', 'label', 'offset')

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.exportLblPng = LblPngExporter()

  @dynamicDocstring(cols=list(allOutputColumns))
  def populateMetadata(self,
                       srcDir: t.Union[FilePath, dict, DirectoryDict] = None,
                       margin=0,
                       marginAsPct=False,
                       labelField='Instance ID',
                       includeCols=allOutputColumns,
                       prioritizeById=False,
                       returnLabelMapping=False,
                       missingOk=False,
                       resizeOpts=None,
                       **kwargs
                       ):
    """
    :param srcDir: Where images corresponding to this dataframe are kept. Source image
      filenames are interpreted relative to this directory if they are not absolute. Alternatively, can be a dict
      of name to np.ndarray image mappings
    :param margin: How much padding to give around each component
    :param marginAsPct: Whether the margin should be a percentage of the component size or
      a raw pixel value.
    :param includeCols: Which columns to include in the export list
      pType: checklist
      limits: {cols}
      value: {cols}
      expanded: False
    :param labelField: See ComponentIO.exportLblPng. This label is provided in the output dataframe
      as well, if specified.
    :param prioritizeById: Since the label image export is only one channel (i.e. grayscale), problems arise when
      there is overlap between components. Which one should be on top? If `prioritizeById` is *True*, higher
      ids are always on top of lower ids. So, if ID 1 is a small component and ID 2 is a larger component completely
      surrounding ID 1, ID 1's export will just look like ID 2's export. If *Fals*, the current component is always
      on top in its label mask. In the case where more than 2 components overlap, the other components are ordered
      by ID. So, in the previous scenario ID 1 will still show up on top of ID 2 in its own exported mask despite being
      a lower ID, but ID 2 will be fully visible in its own export.
    :param missingOk: Whether a missing image is acceptable. When no source image is found
      for an annotation, this will simpy the 'image' output property
    :param resizeOpts: Options for reshaping the output to a uniform size if desired. The following keys may be supplied:

      - ``shape``          : Required. It is the shape that all images will be resized to before
                             being saved. This is useful for neural networks with a fixed input size which forces all
                             inputs to be e.g. 100x100 pixels.
      - ``keepAspectRatio``: default True. Whether to keep the aspect ratio and pad the problematic axis, or
                             to stretch the image to the right fit. I.e. if a component with shape (25, 50) exists, and
                             an export ``shape`` of (25, 25) is specified with ``keepAspectRatio``, the component will
                             be resized to (12, 25) and padded on the top and bottom with 6 and 7 pixels of ``padVal``,
                             respectively.
      - ``padVal``         : default np.nan. How to fill the padded axis if `keepAspectRatio` is *True*.
                             If *np.nan*, the values are grabbed from the image instead. If a component is on the image
                             boundary, black (0) is used.
      - ``allowReorient``  : default False. If *True*, the output image can be rotated 90 degrees if this reduces the
                             amount of manipulation required to get the output to be the proper shape
      - ``interpolation``  : Any interpolation value accepted by cv.resize
    :param returnLabelMapping: Whether to return the mapping of label numeric values to table field values
    :return: Dataframe with the following keys:
      - instId: The component's Instance ID
      - img: The (MxNxC) image corresponding to the component vertices, where MxN are
        the padded row sizes and C is the number of image channels
      - labelMask: Binary mask representing the component vertices
      - label: Field value of the component for the field specified by `labelField`
      - offset: Image (x,y) coordinate of the min component vertex.
    """
    if srcDir is None:
      srcDir = Path('.')
    srcDir = DirectoryDict(srcDir, allowAbsolute=True, readFunc=cvImread_rgb)
    self.srcDir = srcDir
    # Label masks are programmatically generated so no need for a backing directory
    self.labelMaskDir = {}
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

  def bulkExport(self,
                 compDf,
                 exportObj,
                 missingOk=None,
                 **kwargs):
    # imageShape is automatically inferred by the exporter
    kwargs.pop('imageShape', None)
    # File is taken care of in outer scope
    kwargs.pop('file', None)
    mappings = {}

    for fullImgName, miniDf in compDf.groupby(RTF.IMG_FILE): # type: str, pd.DataFrame
      exportedComps, mapping = self._formatSingleImage(miniDf, fullImgName, missingOk, **kwargs)
      mappings[Path(fullImgName).name] = mapping
      exportObj.extend(exportedComps)

    exportObj = pd.DataFrame(exportObj)
    if len(mappings) == 1:
      # Common case where annotations for just one image were converted
      mappings = next(iter(mappings.values()))
    exportObj.attrs['mapping'] = mappings
    self.mappings = mappings
    return exportObj, NO_ERRORS

  def _formatSingleImage(self, compDf: pd.DataFrame, imageName, missingOk=False, **kwargs):
    exportObj = []
    img = self.srcDir.get(imageName)
    if img is None and not missingOk:
      raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), imageName)
    shape = img if img is None else img.shape[:2]
    # Make sure no options are duplicated
    useKwargs = {**kwargs, **dict(returnLabelMapping=True, imageShape=shape)}
    lblImg, mapping = self.exportLblPng(compDf, **useKwargs)
    if img is None:
      img = np.zeros_like(lblImg)
    self.invertedMap = pd.Series(mapping.index, mapping)
    for row in pd_iterdict(compDf):
      exportObj.append(self._formatSingleComp(row, image=img, labelImage=lblImg, **kwargs))

    return exportObj, mapping

  def _formatSingleComp(self,
                        inst: t.Any, *,
                        image=None,
                        labelImage=None,
                        labelField=None,
                        margin=None,
                        marginAsPct=None,
                        includeCols=None,
                        prioritizeById=None,
                        resizeOpts=None,
                        returnStats=None,
                        **_kwargs):
    out = {}
    allVerts = inst[RTF.VERTICES].stack()
    imageName = inst[RTF.IMG_FILE]
    if image is None:
      image = self.srcDir.get(imageName)
    if labelImage is None:
      labelImage = self.labelMaskDir.get(imageName)
    if marginAsPct:
      compImgSz = allVerts.max(0) - allVerts.min(0)
      marginToUse = (compImgSz * (margin / 100)).astype(int)
    else:
      marginToUse = margin

    returnStats = returnStats or resizeOpts.pop('returnStats', None)
    if returnStats:
      compImg, bounds, stats = self.cropperFunc(
        image, allVerts, marginToUse, returnCoords=True, returnStats=returnStats, **resizeOpts
      )
    else:
      compImg, bounds = self.cropperFunc(
        image, allVerts, marginToUse, returnCoords=True, **resizeOpts
      )
      stats = None
    useKeys = includeCols

    if 'instanceId' in useKeys:
      out['instanceId'] = inst[RTF.INST_ID]

    if 'offset' in useKeys:
      out['offset'] = bounds[0, :]

    lbl = inst[labelField]
    if 'label' in useKeys:
      out['label'] = lbl

    if stats is not None:
        out.update(stats)

    if 'image' in useKeys:
      out['image'] = compImg

    if 'labelMask' in useKeys:
      if prioritizeById:
        # ID indicates z-value, which is already the case for a label image
        useImg = labelImage
      else:
        # The current component should always be drawn on top
        useImg = inst[RTF.VERTICES].toMask(labelImage.copy(),
                                          float(self.invertedMap[lbl]),
                                          asBool=False)
      mask = self.cropperFunc(useImg, allVerts, marginToUse, returnCoords=False, **resizeOpts)

      out['labelMask'] = mask

    return out

class CompImgsZipExporter(CompImgsDfExporter):

  def populateMetadata(self,
                       archive=False,
                       makeSummary=False,
                       summaryImageWidth=None,
                       **kwargs):
    """
    :param archive: Whether to compress into a zip archive instead of directly outputting a folder
    :param makeSummary: Whether to include an html table showing each component from the dataframe along with
      its image and mask representations
    :param summaryImageWidth: If `makeSummary` is *True*, this parameter can determine whether each image should have
      a fixed width in the summary table. This can be useful to normalize the display of all images. Should be in pixels
    """
    ret = super().populateMetadata(**kwargs)
    ret.update(self._forwardMetadata(locals()))
    return ret

  def writeFile(self,
                file: FilePath,
                exportObj,
                archive=None,
                makeSummary=None,
                summaryImageWidth=None,
                **kwargs):
    outDir = Path(file)
    useDir = outDir

    with ExitStack() as stack:
      if archive:
        useDir = Path(stack.enter_context(tempfile.TemporaryDirectory()))
      dataDir = useDir / 'data'
      labelsDir = useDir / 'labels'
      dataDir.mkdir(exist_ok=True, parents=True)
      labelsDir.mkdir(exist_ok=True, parents=True)

      summaryName = useDir / 'summary.html'

      for idx, row in exportObj.iterrows():
        saveName = f'{row["instanceId"]}.png'
        if 'image' in row.index:
          cvImsave_rgb(dataDir / saveName, row['image'])
        if 'labelMask' in row.index:
          cvImsave_rgb(labelsDir / saveName, row.labelMask)

      if makeSummary:
        self._createSummary(exportObj, useDir, dataDir, labelsDir, summaryName, summaryImageWidth)
      else:
        summaryName = None

      if archive:
        self._createArchive(outDir, dataDir, labelsDir, summaryName)

  @staticmethod
  def _createArchive(parentDir, dataDir, labelsDir, summaryName):
    makeSummary = summaryName is not None
    if parentDir.suffix != '.zip':
      parentDir = parentDir.with_suffix(parentDir.suffix + '.zip')
    with ZipFile(parentDir, 'w') as ozip:
      for dir_ in labelsDir, dataDir:
        if not dir_.exists():
          continue
        for file in dir_.iterdir():
          ozip.write(file, f'{dir_.name}/{file.name}')
      if makeSummary:
        ozip.write(summaryName, file.name)

  def _createSummary(self, exportObj, parentDir, dataDir, labelsDir, summaryName, imageWidth):
    extractedImgs = exportObj.rename({'instanceId': RTF.INST_ID.name}, axis=1)
    # Prevent merge error by renaming index
    # INST_ID.name has to be used instead of raw INST_ID due to strange pandas issue
    # throwing a TypeError: keywords must be a string
    outDf: pd.DataFrame = self.compDf.drop([RTF.VERTICES], axis=1).rename(str, axis=1)
    outDf = outDf.merge(
      extractedImgs, on=RTF.INST_ID.name
    )
    for colName, imgDir in zip(['labelMask', 'image'], [labelsDir, dataDir]):
      if colName not in extractedImgs:
        continue
      relDir = imgDir.relative_to(parentDir)
      outDf[colName] = outDf[RTF.INST_ID.name].apply(
        lambda el: imgPathtoHtml((relDir / str(el)).with_suffix('.png').as_posix())
      )
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
    pd_toHtmlWithStyle(outDf, summaryName, style, escape=False, index=False)

class SerialExporter(AnnotationExporter):
  """
  The file suffix can be any option supported by a pandas exporter. This can be
  csv, json, feather, etc.
  Note: pickle is a special case. In some cases, it is significantly more benficial
  to export the raw dataframe compared to a serialized version. In these cases, use
  ComponentIO.exportPkl. Otherwise, `pickle` is still a valid option here for a serialized
  format. For a full list of export options, see
  `the documentation`https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html`.
  """

  def createExportObj(self, **kwargs):
    return self.compDf.copy()

  def populateMetadata(self,
                       readonly=True,
                       **kwargs):
    """
    :param readonly: Whether this export should be read-only
    """
    return self._forwardMetadata(locals())

  @classmethod
  def writeFile(cls,
                file: FilePath,
                exportObj,
                readonly=None,
                **kwargs):
    defaultExportParams = {
      'na_rep': 'NaN',
      'float_format': '{:0.10n}',
      'index': False,
    }
    outPath = Path(file)
    outPath.parent.mkdir(exist_ok=True, parents=True)
    exporter = outPath.suffix.lower().replace('.', '')

    defaultExportParams.update(kwargs)
    exportFn = getattr(exportObj, f'to_{exporter}', None)
    if exportFn is None:
      raise ValueError(
        f'Exporter "{exporter}" not recognized. Acceptable options:\n' +
        ', '.join(cls._getPdExporters())
      )

    with np.printoptions(threshold=sys.maxsize):
      exportFn(file, index=False)
    if readonly:
      outPath.chmod(S_IREAD | S_IRGRP | S_IROTH)

  @staticmethod
  def _getPdExporters():
    members = inspect.getmembers(
      pd.DataFrame, lambda meth: inspect.isfunction(meth) and meth.__name__.startswith('to_')
    )
    return [mem[0].replace('to_', '') for mem in members]

  def bulkExport(self,
                 compDf,
                 exportObj,
                 readonly=None,
                 **kwargs):
    allErrs = []
    # TODO: Currently the additional options are causing errors. Find out why and fix
    #  them, since this may be useful if it can be modified
    # Format special columns appropriately
    # Since CSV export significantly modifies the df, make a copy before doing all these
    # operations
    for col in exportObj:
      if not isinstance(col.value, str):
        serial, errs = serialize(col, exportObj[col])
        exportObj[col] = serial.to_numpy()
        allErrs.append(errs)
    allErrs = pd.concat(allErrs, axis=1)
    return exportObj, allErrs

class CsvExporter(SerialExporter): pass

class PklExporter(AnnotationExporter):
  def writeFile(self, file: FilePath, exportObj, **kwargs):
    exportObj.to_pickle(file)

  def createExportObj(self, **kwargs):
    return self.compDf.copy()
