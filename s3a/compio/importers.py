from __future__ import annotations
import ast
import inspect
import json
import typing as t
from datetime import datetime
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
from skimage import draw

from utilitys.typeoverloads import FilePath
from .base import AnnotationImporter
from .helpers import registerIoHandler
from ..constants import REQD_TBL_FIELDS as RTF
from ..generalutils import DirectoryDict, orderContourPts, cvImread_rgb, pd_iterdict
from ..structures import ComplexXYVertices, XYVertices, AnnInstanceError, LabelFieldType

__all__ = ['SerialImporter', 'CsvImporter', 'SuperannotateJsonImporter', 'GeojsonImporter',
           'LblPngImporter', 'PklImporter', 'CompImgsDfImporter', 'VGGImageAnnotatorImporter']

class SerialImporter(AnnotationImporter):
  ioType = 's3a'

  @classmethod
  def readFile(cls, file: FilePath, **kwargs):
    fType = Path(file).suffix.lower().replace('.', '')
    importFn = getattr(pd, f'read_{fType}', None)
    if importFn is None:
      raise ValueError(
        f'File type {fType} cannot be handled by the serial importer.'
        f' Must be one of {",".join(cls._getPdImporters())}'
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
    return pd_iterdict(importObj)

  @staticmethod
  def _getPdImporters():
    members = inspect.getmembers(
      pd.DataFrame, lambda meth: inspect.isfunction(meth) and meth.__name__.startswith('read_')
    )
    return [mem[0].replace('read_', '') for mem in members]

class CsvImporter(SerialImporter):
  # Override to provide custom access to csv file type
  pass

class GeojsonImporter(AnnotationImporter):

  def readFile(self, file: FilePath, **kwargs):
    with open(Path(file), 'r') as ifile:
      return json.load(ifile)

  def getInstances(self, importObj, **kwargs):
    return importObj['features']

  def formatSingleInstance(self, inst, **kwargs):
    return inst['properties']

  @staticmethod
  def parseRegion(geometry):
    geometry = ast.literal_eval(geometry)
    if geometry['type'] == 'Polygon':
      return ComplexXYVertices(geometry['coordinates'], coerceListElements=True)
    else:
      return AnnInstanceError(f'Unrecognized type "{geometry["type"]}"')

  bulkImport = AnnotationImporter.defaultBulkImport

registerIoHandler('geojsonregion',
                  deserialize=GeojsonImporter.parseRegion)

class SuperannotateJsonImporter(AnnotationImporter):
  ioType = 'superannotate'

  def readFile(self, file: FilePath, **kwargs):
    with open(Path(file), 'r') as ifile:
      return json.load(ifile)

  def populateMetadata(self,
                       file: Path=None,
                       srcDir: t.Union[FilePath, dict] = None,
                       imageShape: tuple[int, int] = None,
                       **kwargs):
    if srcDir is None:
      srcDir = file.parent
    srcDir = DirectoryDict(srcDir, readFunc=self.readFile, allowAbsolute=True)
    classes = srcDir.get('classes.json')
    if classes is None and file is not None:
      classes = srcDir.get(file.parent / 'classes' / 'classes.json')
    if classes is not None:
      self.tableData.fieldFromName('className').opts['limits'] = [c['name'] for c in classes]
    meta = self.importObj['metadata']
    if imageShape is None:
      imageShape = (meta.pop('height'), meta.pop('width'))
    return self._forwardMetadata(locals())

  def getInstances(self, importObj, **kwargs):
    return importObj['instances']

  def formatSingleInstance(self,
                           inst,
                           name=None,
                           **kwargs):
    out = {
      RTF.IMG_FILE: name
    }
    verts = self.parseRegion(inst)
    if not isinstance(verts, AnnInstanceError):
      verts = ComplexXYVertices([verts])
    # Need to serialize since wrapper function tries to deserialize
    out[RTF.VERTICES] = verts
    return out

  @staticmethod
  def parseRegion(inst: dict) -> t.Union[XYVertices, AnnInstanceError]:
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
  def parseAttributes(attrs):
    if attrs is None:
      attrs = []
    combined = '\n'.join(a['groupName'] for a in attrs)
    return combined

  @staticmethod
  def parseTime(val):
    parsedTime = datetime.strptime(val, '%Y-%m-%dT%H:%M:%S.%fZ')
    return str(parsedTime)

registerIoHandler('superannattributes',
                  deserialize=SuperannotateJsonImporter.parseAttributes)
registerIoHandler(
  'superanntime',
  deserialize=SuperannotateJsonImporter.parseTime,
)

class VGGImageAnnotatorImporter(CsvImporter):
  ioType = None # Will be auto-assigned in init

  def formatSingleInstance(self, inst, **kwargs):
    out = json.loads(inst['region_attributes'])
    return out

  def getInstances(self, importObj: pd.DataFrame, **kwargs):
    return pd_iterdict(importObj)

  @staticmethod
  def parseRegion(region):
    region = json.loads(region)
    if not region:
      return ComplexXYVertices()
    name = region['name']
    if name in ['polygon', 'polyline']:
      pts = XYVertices(np.column_stack((region['all_points_x'], region['all_points_y'])))
      if name == 'polyline':
        pts.connected = False
    elif name == 'ellipse':
      vals = region['cy'], region['cx'], region['ry'], region['rx'], np.rad2deg(region['theta']) + 90
      pts = draw.ellipse_perimeter(*(int(v) for v in vals[:-1]), vals[-1])
      pts = np.column_stack(pts[::-1])
      pts = orderContourPts(pts)
    elif name == 'rect':
      x, y = region['x'], region['y']
      width, height = region['width'], region['height']
      pts = XYVertices([[x, y], [x + width, y], [x + width, y + height], [x, y + height]])
    elif name == 'circle':
      cx, cy, r = region['cx'], region['cy'], region['r']
      pts = draw.circle_perimeter(int(cy), int(cx), int(r))
      pts = np.column_stack(pts[::-1])
      pts = orderContourPts(pts)
    elif name == 'point':
      cx, cy = region['cx'], region['cy']
      pts = XYVertices([[cx, cy]])
    else:
      raise ValueError(f'Unknown region shape: "{name}')
    return ComplexXYVertices([pts])

registerIoHandler('viaregion', deserialize=VGGImageAnnotatorImporter.parseRegion)

class LblPngImporter(AnnotationImporter):
  ioType = 's3a'

  imgInfo = {}

  def readFile(self, file: FilePath,
               labelMapping=None,
               offset=0,
               **kwargs):
    try:
      image: Image.Image = Image.open(file)
      self.imgInfo = image.info
      # False positive
      # noinspection PyTypeChecker
      image = np.asarray(image)
    except TypeError:
      # E.g. float image
      return cvImread_rgb(str(file), mode=cv.IMREAD_UNCHANGED)
    return image

  def populateMetadata(self,
                       labelField: LabelFieldType = 'Instance ID',
                       labelMapping: pd.Series = None,
                       distinctRegions=True,
                       offset=0,
                       **kwargs):
    """
    :param labelField: label field to associate with this image. Pixels values within the image
      correspond to values from this field in the table data. If *None*, this is inferred by the mapping read
      from the image file (see `labelMapping` description)
    :param labelMapping: For parameters that aren't numeric and don't have limits (e.g. arbitrary string values),
      this mapping determines how numeric values should be turned into field values. See `PrjParam.toNumeric` for
      details, since this is the mapping expected. If not provided, first the image metadata tags are searched for
      a 'labelMapping' text attribute (this is often added to label images saved by S3A). Note that metadata can only be
      read from the file if a file path is provided, of course. If this check fails, it is inferred based on the
      allowed options of `labelField` (`labelField.opts['limits']`). Finally, if this is not present, it is assumed the
      raw image values can be used directly as field values.
    :param offset: When `labelMapping` is not provided and field values are directly inferred from label values, this
      determines whether (and how much if not *None*) to offset numeric labels during import. I.e. if the png label
      is 1, but offset is 1, the corresponding *field* value will be 0 (1 - offset = 0).
    :param distinctRegions: Whether separate regions with the same ID should be separate IDs, or
      one ID with a group of polygons
    """
    # Rename for clarity
    labelImage = self.importObj
    # "Offset" present for numeric data, "mapping" present for textual data
    info = self.imgInfo
    if labelMapping is None and 'labelMapping' in info:
      labelMapping = pd.Series(json.loads(info['labelMapping']), name=info.get('labelField', None))
      labelMapping.index = labelMapping.index.astype(int)

    if offset is None and 'offset' in info:
      offset = int(info['offset'])

    labelField = self.tableData.fieldFromName(labelField or labelMapping.name)
    if labelMapping is None:
      vals = labelField.opts.get('limits', None) or np.unique(labelImage)
      _, labelMapping = labelField.toNumeric(vals, returnMapping=True)
      labelMapping.index += offset

    return self._forwardMetadata(locals())

  def getInstances(self,
                   importObj,
                   labelMapping=None,
                   distinctRegions=None,
                   **kwargs):
    labelMask = importObj
    for numericLbl, origVal in labelMapping.iteritems(): # type: int, t.Any
      verts = ComplexXYVertices.fromBinaryMask(labelMask == numericLbl)
      if distinctRegions:
        for vv in verts:
          yield {RTF.INST_ID: numericLbl,
                 RTF.VERTICES: ComplexXYVertices([vv])}
      else:
        yield {RTF.INST_ID: numericLbl,
               RTF.VERTICES: verts}

  bulkImport = AnnotationImporter.defaultBulkImport

class PklImporter(AnnotationImporter):
  ioType = 's3a'

  def readFile(self, file: FilePath, **importArgs) -> pd.DataFrame:
    """
    See docstring for :func:`self.importCsv`
    """
    return pd.read_pickle(file)

  def getInstances(self, importObj, **kwargs):
    return pd_iterdict(importObj)

  def bulkImport(self, importObj, **kwargs):
    return importObj

class CompImgsDfImporter(AnnotationImporter):
  ioType = 's3a'

  readFile = PklImporter.readFile

  def getInstances(self, importObj, **kwargs):
    return importObj.iterrows()

  def populateMetadata(self,
                       labelField: LabelFieldType = 'Instance ID',
                       **kwargs
                       ):
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

  def bulkImport(self,
                 importObj,
                 labelField=None,
                 **kwargs):
    out = importObj[['instanceId', 'label']].copy()
    out.columns = [RTF.INST_ID, labelField]
    return out
