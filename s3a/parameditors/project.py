import shutil
from pathlib import Path
from typing import List, Optional, Union, Set, Dict

from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
from skimage import io
import pandas as pd

from s3a.constants import BASE_DIR, REQD_TBL_FIELDS
from s3a.generalutils import resolveYamlDict
from s3a.graphicsutils import saveToFile, popupFilePicker, PopupLineEditor
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

class ThumbnailViewer(QtWidgets.QListWidget):
  sigImageSelected = QtCore.Signal(object) # Image full path to delete
  def __init__(self, parent=None):
    super().__init__(parent)
    self.nameToFullPathMapping: Dict[str, Path] = {}
    self.setViewMode(self.IconMode)
    self.setIconSize(QtCore.QSize(200,200))
    self.setResizeMode(self.Adjust)

  def addThumbnail(self, fullName: Path):
    icon = QtGui.QIcon(str(fullName))
    newItem = QtWidgets.QListWidgetItem(fullName.name)
    newItem.setIcon(icon)
    self.addItem(newItem)
    self.nameToFullPathMapping[fullName.name] = fullName

class ProjectData:
  def __init__(self):
    self.tableData = TableData()
    self.cfg = {}
    self.cfgFname: Optional[Path] = Path()
    self.images: List[Path] = []
    self.baseImgDirs: Set[Path] = set()
    self.imgToAnnMapping: Dict[Path, Path] = {}
    """Records annotations belonging to each image"""
    self.exportOpts = {}

    self.compIo = ComponentIO()
    self.compIo.tableData = self.tableData

    self._thumbnails = ThumbnailViewer()

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
        image = Path(image)
        if image.is_dir():
          self.addImageFolder(image, shouldCopy)
          continue
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
    for folder in self.baseImgDirs:
      if location in folder.parents:
        folder = folder.relative_to(location)
      strImgNames.append(str(folder))
    for img in self.images:
      if img.parent in self.baseImgDirs:
          # This image is already accounted for in the base directories
          continue
      strImgNames.append(str(img))
    for ann in self.imgToAnnMapping.values():
      if location in ann.parents:
        outName = str(ann.relative_to(location))
      else:
        outName = str(ann)
      strAnnNames.append(outName)
    self.cfg['images'] = strImgNames
    self.cfg['annotations'] = strAnnNames
    saveToFile(self.cfg, self.cfgFname)

  def addImage(self, name: FilePath, data: NChanImg=None, copyToProj=False):
    name = Path(name).resolve()
    if copyToProj or data is not None:
      name = self._copyImgToProj(name, data)
    if name not in self.images:
      self.images.append(name)
    self._thumbnails.addThumbnail(name)

  def addImageFolder(self, folder: FilePath, copyToProj=False):
    folder = Path(folder)
    if copyToProj:
      newFolder = self.imagesDir/folder.name
      shutil.copytree(folder, newFolder)
      folder = newFolder
      copyToProj = False
    self.baseImgDirs.add(folder)
    for img in folder.glob('*.*'):
      self.addImage(img, copyToProj=copyToProj)

  def addImage_gui(self, copyToProject=True):
    fileFilter = "Image Files (*.png *.tif *.jpg *.jpeg *.bmp *.jfif);;All files(*.*)"
    fname = popupFilePicker(self, 'Add Image to Project', fileFilter)
    if fname is not None:
      self.addImage(fname, copyToProj=copyToProject)

  def removeImage(self, imgName: FilePath):
    imgName = Path(imgName).resolve()
    self.images.remove(imgName)
    # Remove copied annotations for this image
    for ann in self.annotationsDir.glob(f'{imgName.stem}.*'):
      ann.unlink()
    self.imgToAnnMapping.pop(imgName, None)

  def removeImage_gui(self):
    self._thumbnails.show()
    self._thumbnails.exec_()


  def removeAnnotation(self, annName: FilePath):
    annName = Path(annName).resolve()
    # Since no mapping exists of all annotations, loop the long way until the file is found
    for key, ann in self.imgToAnnMapping.items():
      if annName == ann:
        del self.imgToAnnMapping[key]
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
    image = Path(image).resolve()
    # Since only one annotation file can exist per image, concatenate this with any existing files for the same image
    # if needed
    annForImg = self.imgToAnnMapping.get(image, None)
    oldAnns = ComponentIO.buildByFileType(annForImg)
    combinedAnns = oldAnns + [data]
    outAnn = pd.concat(combinedAnns, ignore_index=True)
    outAnn[REQD_TBL_FIELDS.INST_ID] = outAnn.index
    outFmt = f".{self.exportOpts['annotation-format']}"
    outName = self.annotationsDir / image.with_suffix(outFmt).name
    ComponentIO.exportByFileType(outAnn, outName, verifyIntegrity=False, readOnly=False, imgDir=self.imagesDir)
    self.imgToAnnMapping[image] = outName

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
    return newName.resolve()

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