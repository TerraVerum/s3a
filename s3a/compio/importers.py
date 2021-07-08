import inspect
import json
import typing as t
from collections import defaultdict
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
from skimage import draw

from utilitys.typeoverloads import FilePath
from .base import AnnotationImporter
from .helpers import _getPdImporters
from ..constants import REQD_TBL_FIELDS as RTF
from ..generalutils import DirectoryDict, orderContourPts, cvImread_rgb, deprecateKwargs
from ..structures import ComplexXYVertices, XYVertices, AnnInstanceError, LabelFieldType

__all__ = ['SerialImporter', 'CsvImporter', 'SuperannotateJsonImporter', 'GeojsonImporter',
           'LblPngImporter', 'PklImporter', 'CompImgsDfImporter']

class SerialImporter(AnnotationImporter):

  def readFile(self, filename: FilePath, **kwargs):
    fType = Path(filename).suffix.lower().replace('.', '')
    importFn = getattr(pd, f'read_{fType}', None)
    if importFn is None:
      raise ValueError(
        f'File type {fType} cannot be handled by the serial importer.'
        f' Must be one of {",".join(_getPdImporters())}'
      )
    # Special case: csv imports need to avoid interpreting nan results
    kwargs.update(na_filter=False, dtype=str)
    acceptedArgs = inspect.signature(importFn).parameters
    useArgs = kwargs.keys() & acceptedArgs
    serialDf = importFn(filename, **{k: kwargs[k] for k in useArgs})
    return serialDf

  def bulkImport(self, importObj, errorOk=False, **kwargs):
    return importObj

class CsvImporter(SerialImporter):
  # Override to provide custom access to csv file type
  pass

class GeojsonImporter(AnnotationImporter):
  def readFile(self, filename: FilePath, **kwargs):
    with open(Path(filename), 'r') as ifile:
      return json.load(ifile)

  def getInstances(self, importObj):
    return importObj['features']

  def formatSingleInstance(self, inst, **kwargs):
    geo = inst['geometry']
    out = {}
    if geo['type'] == 'Polygon':
      out[RTF.VERTICES] = ComplexXYVertices(geo['coordinates'], coerceListElements=True)
    else:
      out[RTF.VERTICES] = AnnInstanceError(f'Unrecognized type "{geo["type"]}"')
    return out

class SuperannotateJsonImporter(AnnotationImporter):
  def readFile(self, filename: FilePath, **kwargs):
    with open(Path(filename), 'r') as ifile:
      return json.load(ifile)

  def populateMetadata(self, filename: FilePath=None, srcDir: t.Union[FilePath, dict] = None, **kwargs):
    if srcDir is None:
      srcDir = filename.parent
    srcDir = DirectoryDict(srcDir, readFunc=self.readFile, allowAbsolute=True)
    classes = srcDir.get('classes.json')
    if classes is None and filename is not None:
      classes = srcDir.get(filename.parent / 'classes' / 'classes.json')
    if classes is not None:
      self.tableData.fieldFromName('className').opts['limits'] = [c['name'] for c in classes]
    self.opts = self.importObj['metadata']
    self.opts['imShape'] = (self.opts['height'], self.opts['width'])

  def getInstances(self, importObj):
    return importObj['instances']

  def formatSingleInstance(self, inst, **kwargs):
    out = {
      RTF.SRC_IMG_FILENAME: self.opts['name']
    }
    verts = self._parsePts(inst)
    if not isinstance(verts, AnnInstanceError):
      verts = ComplexXYVertices([verts])
    out.update(self._parseAttrs(inst))
    # Need to serialize since wrapper function tries to deserialize
    out[RTF.VERTICES] = verts
    return out

  @staticmethod
  def _parsePts(inst: dict) -> t.Union[XYVertices, AnnInstanceError]:
    typ = inst['type']
    if typ == 'polygon':
      pts = inst['points']
      pts = np.column_stack([pts[::2], pts[1::2]])
    elif typ in ('bbox', 'rbbox'):
      dictPts = inst['points']
      def keys():
        ii = 1
        while True:
          yield f'x{ii}'
          yield f'y{ii}'
          ii += 1
      pts = []
      for kk in keys():
        if kk not in dictPts:
          break
        pts.append(dictPts[kk])
      # x-y list is not formatted like a box -- fix this
      tempVerts = np.column_stack([pts[::2], pts[1::2]])
      wh = np.diff(tempVerts, axis=0)
      if len(tempVerts) == 2:
        pts = wh * [[0, 0], [1, 0], [1, 1], [0, 1]] + tempVerts[0]
      else:
        pts = tempVerts
    elif typ == 'ellipse':
      vals = inst['cy'], inst['cx'], inst['ry'], inst['rx'], inst['angle']
      pts = draw.ellipse_perimeter(*(int(v) for v in vals))
      pts = np.column_stack(pts[::-1])
      pts = orderContourPts(pts)
    else:
      pts = AnnInstanceError(f'Unrecognized type "{typ}"')
    if not isinstance(pts, AnnInstanceError):
      pts = pts.view(XYVertices)
    return pts

  @staticmethod
  def _parseAttrs(inst):
    attrs = inst.get('attributes', [])
    notes = '\n'.join(a['groupName'] for a in attrs)
    return {'Notes': notes}

class LblPngImporter(AnnotationImporter):
  imgInfo = {}
  _canBulkImport = False

  def readFile(self, filename: FilePath,
               labelMapping=None,
               offset=0,
               **kwargs):
    try:
      image: Image.Image = Image.open(filename)
      self.imgInfo = image.info
      # False positive
      # noinspection PyTypeChecker
      image = np.asarray(image)
    except TypeError:
      # E.g. float image
      return cvImread_rgb(str(filename), mode=cv.IMREAD_UNCHANGED)
    return image

  @deprecateKwargs(lblMapping='labelMapping', lblField='labelField')
  def populateMetadata(self,
                       labelField: LabelFieldType = 'Instance ID',
                       labelMapping: pd.Series = None,
                       distinctRegions=True,
                       offset=0,
                       **kwargs):
    """
    :param labelField: label field to associate with this image. Pixels values within the image
      correspond to values from this field in the table data. If *None*, this is inferred by the mapping read
      from the image file (see `lblMapping` description)
    :param labelMapping: For parameters that aren't numeric and don't have limits (e.g. arbitrary string values),
      this mapping determines how numeric values should be turned into field values. See `PrjParam.toNumeric` for
      details, since this is the mapping expected. If not provided, first the image metadata tags are searched for
      a 'lblMapping' text attribute (this is often added to label images saved by S3A). Note that metadata can only be
      read from the file if a file path is provided, of course. If this check fails, it is inferred based on the
      allowed options of `lblField` (`lblField.opts['limits']`). Finally, if this is not present, it is assumed the
      raw image values can be used directly as field values.
    :param offset: When `lblMapping` is not provided and field values are directly inferred from label values, this
      determines whether (and how much if not *None*) to offset numeric labels during import. I.e. if the png label
      is 1, but offset is 1, the corresponding *field* value will be 0 (1 - offset = 0).
    :param distinctRegions: Whether separate regions with the same ID should be separate IDs, or
      one ID with a group of polygons
    """
    # Rename for clarity
    labelImage = self.importObj
    # "Offset" present for numeric data, "mapping" present for textual data
    info = self.imgInfo
    if labelMapping is None and 'mapping' in info:
      labelMapping = pd.Series(json.loads(info['mapping']), name=info.get('field', None))
      labelMapping.index = labelMapping.index.astype(int)

    if offset is None and 'offset' in info:
      offset = int(info['offset'])

    labelField = self.tableData.fieldFromName(labelField or labelMapping.name)
    if labelMapping is None:
      vals = labelField.opts.get('limits', None) or np.unique(labelImage)
      _, labelMapping = labelField.toNumeric(vals, returnMapping=True)
      labelMapping.index += offset

    self.opts['labelField'] = labelField
    self.opts['labelMapping'] = labelMapping
    self.opts['distinctRegions'] = distinctRegions
    self.opts['offset'] = offset

  def getInstances(self, importObj, **kwargs):
    return self.opts['labelMapping'].iteritems()

  def bulkImport(self, importObj, errorOk=False, **kwargs):
    # Not possible to bulk import
    return pd.DataFrame()

  def formatSingleInstance(self, inst, **kwargs) -> dict:
    numericLbl, origVal = inst
    labelMask: np.ndarray = self.importObj
    verts = ComplexXYVertices.fromBwMask(labelMask == numericLbl)
    out = defaultdict(list)
    if self.opts['distinctRegions']:
      newRegions = [ComplexXYVertices([v]) for v in verts]
      out[RTF.VERTICES].extend(newRegions)
      orig = np.tile(origVal, len(verts))
    else:
      out[RTF.VERTICES].append(verts)
      orig = [origVal]
    out[self.opts['labelField']].extend(orig)
    return out

  def individualImport(self, importObj, **kwargs):
    parsed = defaultdict(list)
    for ii, inst in enumerate(self.getInstances(importObj)):
      parsedInst = self.formatSingleInstance(inst, **kwargs)
      for kk, vv in parsedInst.items():
        parsed[kk].extend(vv)

    return pd.DataFrame(parsed)

class PklImporter(AnnotationImporter):
  def readFile(self, filename: FilePath, **importArgs) -> pd.DataFrame:
    """
    See docstring for :func:`self.importCsv`
    """
    return pd.read_pickle(filename)

  def bulkImport(self, importObj, errorOk=False, **kwargs):
    return self.importObj

class CompImgsDfImporter(AnnotationImporter):

  readFile = PklImporter.readFile

  def getInstances(self, importObj):
    return importObj.iterrows()

  def populateMetadata(self,
                       labelField: LabelFieldType = 'Instance ID',
                       **kwargs
                       ):
    super().populateMetadata(**kwargs)
    self.opts['labelField'] = self.tableData.fieldFromName(labelField)

  def formatSingleInstance(self, inst, **kwargs) -> dict:
    idx, row = inst
    out = {}
    mask = row.labelMask
    verts = ComplexXYVertices.fromBwMask(mask).remove(-row.offset)
    out[RTF.VERTICES] = verts
    return out

  def bulkImport(self, importObj, errorOk=False, **kwargs):
    out = importObj[['instId', 'label']].copy()
    out.columns = [RTF.INST_ID, self.opts['labelField']]
    return out

