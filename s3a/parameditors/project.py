import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Union, Set, Dict

from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
from skimage import io
import pandas as pd

from s3a.constants import BASE_DIR, REQD_TBL_FIELDS
from s3a.generalutils import resolveYamlDict
from s3a.graphicsutils import saveToFile
from s3a.parameditors.table import TableData
from s3a.structures import FilePath, NChanImg, S3AIOError
from s3a.io import ComponentIO

def hierarchicalUpdate(curDict: dict, other: dict):
  """Dictionary update that allows nested keys to be updated without deleting the non-updated keys"""
  for k, v in other.items():
    curVal = curDict.get(k, None)
    if isinstance(curVal, dict):
      hierarchicalUpdate(curVal, v)
    else:
      curDict[k] = v

class ProjectData:
  def __init__(self):
    self.tableData = TableData()
    self.cfg = {}
    self.cfgFname: Optional[Path] = Path()
    self.images: Set[Path] = set()
    self.imgToAnnMapping: Dict[Path, List[Path]] = {}
    """Records annotations belonging to each image"""
    self.exportOpts = {}

    self.compIo = ComponentIO()
    self.compIo.tableData = self.tableData

    self._thumbnails = QtWidgets.QListWidget()
    self._thumbnails.setViewMode(self._thumbnails.IconMode)
    self._thumbnails.setIconSize(QtCore.QSize(200,200))
    self._thumbnails.setResizeMode(self._thumbnails.Adjust)

  @property
  def location(self):
      return self.cfgFname.parent
  @property
  def imagesDir(self):
      return self.location/'images'
  @property
  def annotationsDir(self):
      return self.location/'annotations'


  def loadCfg(self, cfgFname: FilePath=None, cfgDict: dict = None):
    _, defaultCfg = resolveYamlDict(BASE_DIR/'projectcfg.yml')
    cfgFname, cfgDict = resolveYamlDict(cfgFname, cfgDict)
    hierarchicalUpdate(defaultCfg, cfgDict)
    cfg = self.cfg = defaultCfg
    self.exportOpts = cfg['export-opts']
    self.cfgFname = cfgFname
    tableInfo = cfg.get('table-cfg', {})
    if isinstance(tableInfo, str):
      tableDict = None
      tableName = tableInfo
    else:
      tableDict = tableInfo
      tableName = cfgFname
    self.tableData.loadCfg(tableName, tableDict)

  def createProject(self, *, name: FilePath= './projectcfg.yml', cfg: dict=None):
    """
    Creates a new project with the specified settings in the specified directory.
    :param name:
      helpText: Project Name. The parent directory of this name indicates the directory in which to create the project
      pType: filepicker
    :param cfg: see `ProjectData.loadCfg` for information
    """
    name = Path(name)
    location = name.parent
    location = Path(location)
    location.mkdir(exist_ok=True, parents=True)

    if not name.exists() and cfg is None:
      cfg = {}
    self.loadCfg(name, cfg)

    self.annotationsDir.mkdir(exist_ok=True)
    self.imagesDir.mkdir(exist_ok=True)

    shouldCopy = self.cfg['import-opts']['copy-data']
    for image in self.cfg['images']:
      if not isinstance(image, dict):
        image = {'name': image}
      self.addImage(**image, copyToProj=shouldCopy)
    for annotation in self.cfg['annotations']:
      if not isinstance(annotation, dict):
        annotation = {'name': annotation}
      self.addAnnotation(**annotation)

    newName = self.tableData.cfgFname.name
    saveToFile(self.tableData.cfg, location/newName, True)
    self.tableData.cfgFname = newName
    self.cfg['table-cfg'] = newName

    self.saveCfg()

  def saveCfg(self):
    strImgNames = []
    strAnnNames = []
    location = self.location
    for img in self.images:
      if location in img.parents:
        outName = str(img.relative_to(location))
      else:
        outName = str(img)
      strImgNames.append(outName)
    for annList in self.imgToAnnMapping.values():
      for ann in annList:
        if location in ann.parents:
          outName = str(ann.relative_to(location))
        else:
          outName = str(ann)
        strAnnNames.append(outName)
    self.cfg['images'] = strImgNames
    self.cfg['annotations'] = strAnnNames
    saveToFile(self.cfg, self.cfgFname)

  def addImage(self, name: FilePath, data: NChanImg=None, copyToProj=False):
    name = Path(name)
    fullName = name.absolute()
    if copyToProj or data is not None:
      name = self._copyImgToProj(name, data)
    self.images.add(name)
    icon = QtGui.QIcon(str(fullName))
    newItem = QtWidgets.QListWidgetItem(fullName.name)
    newItem.setIcon(icon)
    self._thumbnails.addItem(newItem)

  def removeImage(self, imgName: FilePath):
    imgName = Path(imgName).absolute()
    self.images.discard(imgName)
    # Remove copied annotations for this image
    for ann in self.annotationsDir.glob(f'{imgName.stem}.*'):
      ann.unlink()
    self.imgToAnnMapping.pop(imgName, None)

  def removeAnnotation(self, annName: FilePath):
    annName = Path(annName).absolute()
    # Since no mapping exists of all annotations, loop the long way until the file is found
    for annList in self.imgToAnnMapping.values():
      if annName in annList:
        annList.remove(annName)
        break

  def addAnnotation(self, name: FilePath=None, data: pd.DataFrame=None, image: FilePath=None):
    # Housekeeping for default arguments
    if name is None and data is None:
      raise S3AIOError('`name` and `data` cannot both be `None`')
    elif name is None and image is None:
      raise S3AIOError('`name` and `image` cannot both be `None`')
    if data is None:
      data = ComponentIO.buildByFileType(name)
    if image is None:
      # If no explicit matching to an image is provided, try to determine based on annotation name
      image = self._getImgForAnn(Path(name))
    image = Path(image).absolute()
    # Since only one annotation file can exist per image, concatenate this with any existing files for the same image
    # if needed
    annsForImg = self.imgToAnnMapping.get(image, {})
    # For now this is a method of batch-loading instead of forcing individual file read and appends
    oldAnns = [ComponentIO.buildByFileType(ann) for ann in annsForImg]
    combinedAnns = oldAnns + [data]
    outAnn = pd.concat(combinedAnns, ignore_index=True)
    outAnn[REQD_TBL_FIELDS.INST_ID] = outAnn.index
    outFmt = f".{self.exportOpts['annotation-format']}"
    outName = self.annotationsDir / image.with_suffix(outFmt).name
    ComponentIO.exportByFileType(outAnn, outName, verifyIntegrity=False, readOnly=False, imgDir=self.imagesDir)
    annsForImg[image] = outName

  def _copyImgToProj(self, name: Path, data: NChanImg=None):
    newName = self.imagesDir/name.name
    if name.exists() and data is None:
      shutil.copy(name, newName)
    elif data is not None:
      # Programmatically created or not from a local file
      # noinspection PyTypeChecker
      io.imsave(newName, data)
    else:
      raise S3AIOError(f'No image data associated with {name.name}. Either the file does not exist or no'
                       f' image information was provided.')
    return newName.absolute()

  def _getImgForAnn(self, name: Path):
    matchName = name.stem
    candidates = []
    for imgName in self.images:
      if imgName.stem == matchName:
        candidates.append(imgName)
    numCandidates = len(candidates)
    if numCandidates != 1:
      msg = f'Annotation was added without an explicit mapping to an image. This is only allowed when' \
            f' exactly one corresponding image file has a similar name. Encountered {numCandidates} images' \
            f' with similar names'
      if numCandidates == 0:
        msg += '.'
      else:
        msg += f':\n{", ".join([c.name for c in candidates])}'
      raise S3AIOError(msg)
    else:
      return candidates[0]