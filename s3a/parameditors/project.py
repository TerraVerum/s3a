from __future__ import annotations
import shutil
from pathlib import Path
from typing import List, Optional, Set, Dict
from warnings import warn

import numpy as np
import pandas as pd
from skimage import io

from s3a.constants import BASE_DIR, REQD_TBL_FIELDS, PROJ_FILE_TYPE
from s3a.generalutils import resolveYamlDict, dynamicDocstring
from s3a.graphicsutils import saveToFile, popupFilePicker
from s3a.io import ComponentIO
from s3a.parameditors.table import TableData
from s3a.structures import FilePath, NChanImg, S3AIOError, S3AWarning


def hierarchicalUpdate(curDict: dict, other: dict):
  """Dictionary update that allows nested keys to be updated without deleting the non-updated keys"""
  if other is None:
    return
  for k, v in other.items():
    curVal = curDict.get(k, None)
    if isinstance(curVal, dict) and isinstance(v, dict):
      hierarchicalUpdate(curVal, v)
    else:
      curDict[k] = v

class ProjectData:
  def __init__(self, cfgFname: FilePath=None, cfgDict: dict=None):
    self.tableData = TableData()
    self.cfg = {}
    self.cfgFname: Optional[Path] = None
    self.images: List[Path] = []
    self.baseImgDirs: Set[Path] = set()
    self.imgToAnnMapping: Dict[Path, Path] = {}
    """Records annotations belonging to each image"""

    self.compIo = ComponentIO()
    self.compIo.tableData = self.tableData

    if cfgFname is not None or cfgDict is not None:
      self.loadCfg(cfgFname, cfgDict)

  @property
  def location(self):
      return self.cfgFname.parent
  @property
  def imagesDir(self):
      return self.location/'images'
  @property
  def annotationsDir(self):
      return self.location/'annotations'
  @property
  def settings(self):
    return self.cfg['settings']

  def clearImgs_anns(self):
    for lst in self.images, self.imgToAnnMapping, self.baseImgDirs:
      lst.clear()

  def loadCfg(self, cfgFname: FilePath=None, cfgDict: dict = None):
    self.clearImgs_anns()
    _, defaultCfg = resolveYamlDict(BASE_DIR/'projectcfg.yml')
    cfgFname, cfgDict = resolveYamlDict(cfgFname, cfgDict)
    hierarchicalUpdate(defaultCfg, cfgDict)
    cfg = self.cfg = defaultCfg
    self.cfgFname = cfgFname.resolve()
    tableInfo = cfg.get('table-cfg', None)
    if isinstance(tableInfo, str):
      tableDict = None
      tableName = tableInfo
    else:
      if tableInfo is None:
        tableInfo = {}
      tableDict = tableInfo
      tableName = cfgFname
    tableName = Path(tableName)
    if not tableName.is_absolute():
      tableName = self.location/tableName
    self.tableData.loadCfg(tableName, tableDict)

    self.addImageFolder(self.imagesDir)
    for image in self.cfg['images']:
      if isinstance(image, dict):
        image.setdefault('copyToProj', False)
        self.addImage(**image)
      else:
        self.addImageByPath(image, False)
    for annotation in self.cfg['annotations']:
      if isinstance(annotation, dict):
        self.addAnnotation(**annotation)
      else:
        self.addAnnotationByPath(annotation)

  @classmethod
  def create(cls, *, name: FilePath= f'./{PROJ_FILE_TYPE}', cfg: dict=None, parent: ProjectData=None):
    """
    Creates a new project with the specified settings in the specified directory.
    :param name:
      helpText: Project Name. The parent directory of this name indicates the directory in which to create the project
      pType: filepicker
    :param cfg: see `ProjectData.loadCfg` for information
    """
    name = Path(name)
    name = name/f'{name.name}.{PROJ_FILE_TYPE}'
    location = name.parent
    location.mkdir(exist_ok=True, parents=True)
    if parent is None:
      parent = cls()
    parent.cfgFname = name.resolve()
    parent.annotationsDir.mkdir(exist_ok=True)
    parent.imagesDir.mkdir(exist_ok=True)

    if not name.exists() and cfg is None:
      cfg = {}
    parent.loadCfg(name, cfg)

    tdName = parent.tableData.cfgFname
    if tdName.resolve() != parent.cfgFname:
      tdName = tdName.name
      saveToFile(parent.tableData.cfg, location / tdName, True)
      parent.tableData.cfgFname = tdName
    else:
      tdName = tdName.name
    parent.cfg['table-cfg'] = tdName

    parent.saveCfg()
    return parent

  @classmethod
  def open(cls, name: FilePath):
    proj = cls()
    proj.loadCfg(name)

  def saveCfg(self):
    location = self.location
    annDir = self.annotationsDir
    strAnnNames = [str(annDir.relative_to(location))]
    strImgNames = []
    for folder in self.baseImgDirs:
      if location in folder.parents:
        folder = folder.relative_to(location)
      strImgNames.append(str(folder))
    for img in self.images:
      if img.parent in self.baseImgDirs:
        # This image is already accounted for in the base directories
        continue
      strImgNames.append(str(img.resolve()))

    offendingAnns = []
    for img, ann in self.imgToAnnMapping.items():
      if ann.parent != annDir:
        offendingAnns.append(str(ann))
    if len(offendingAnns) > 0:
      warn('Encountered annotation(s) in project config, but not officially added. Offending files:\n'
           + ',\n'.join(offendingAnns), S3AWarning)
    self.cfg['images'] = strImgNames
    self.cfg['annotations'] = strAnnNames
    saveToFile(self.cfg, self.cfgFname)

  def addImageByPath(self, name: FilePath, copyToProj=False):
    """Determines whether to add as a folder or file based on filepath type"""
    image = Path(name)
    if not image.is_absolute():
      image = self.location/image
    if not image.exists():
      warn(f'Provided image path does not exist: {image}\nNo action performed.', S3AWarning)
      return
    if image.is_dir():
      self.addImageFolder(image, copyToProj)
    else:
      self.addImage(name, copyToProj=copyToProj)

  def addImage(self, name: FilePath, data: NChanImg=None, copyToProj=False, allowOverwrite=False):
    name = Path(name).resolve()
    if copyToProj or data is not None:
      name = self._copyImgToProj(name, data, allowOverwrite)
    if name not in self.images:
      self.images.append(name)
    return name

  def changeImgPath(self, oldName: Path, newName: Path=None):
    oldIdx = self.images.index(oldName)
    if newName is None or newName in self.images:
      del self.images[oldIdx]
    else:
      self.images[oldIdx] = newName

  def addImageFolder(self, folder: FilePath, copyToProj=False):
    folder = Path(folder).resolve()
    if folder in self.baseImgDirs:
      return
    if copyToProj:
      newFolder = self.imagesDir/folder.name
      shutil.copytree(folder, newFolder)
      folder = newFolder
      copyToProj = False
    self.baseImgDirs.add(folder)
    for img in folder.glob('*.*'):
      self.addImage(img, copyToProj=copyToProj)

  def addAnnotationByPath(self, name: FilePath):
    """Determines whether to add as a folder or file based on filepath type"""
    name = Path(name)
    if not name.is_absolute():
      name = self.location/name
    if not name.exists():
      warn(f'Provided annotation path does not exist: {name}\nNo action performed.', S3AWarning)
      return
    if name.is_dir():
      self.addAnnotationFolder(name)
    else:
      self.addAnnotation(name)

  def addAnnotationFolder(self, folder: FilePath, okAddSelf=False):
    folder = Path(folder).resolve()
    for file in folder.glob('*.*'):
      self.addAnnotation(file)

  def addImage_gui(self, copyToProject=True):
    fileFilter = "Image Files (*.png *.tif *.jpg *.jpeg *.bmp *.jfif);;All files(*.*)"
    fname = popupFilePicker(None, 'Add Image to Project', fileFilter)
    if fname is not None:
      self.addImage(fname, copyToProj=copyToProject)

  def removeImage(self, imgName: FilePath):
    imgName = Path(imgName).resolve()
    self.images.remove(imgName)
    # Remove copied annotations for this image
    for ann in self.annotationsDir.glob(f'{imgName.stem}.*'):
      ann.unlink()
    self.imgToAnnMapping.pop(imgName, None)

  def removeAnnotation(self, annName: FilePath):
    annName = Path(annName).resolve()
    # Since no mapping exists of all annotations, loop the long way until the file is found
    for key, ann in self.imgToAnnMapping.items():
      if annName == ann:
        del self.imgToAnnMapping[key]
        break

  def addAnnotation(self, name: FilePath=None, data: pd.DataFrame=None, image: FilePath=None,
                    overwriteOld=False):
    # Housekeeping for default arguments
    if name is None and data is None:
      raise S3AIOError('`name` and `data` cannot both be `None`')
    if data is None:
      data = ComponentIO.buildByFileType(name)
    if image is None:
      # If no explicit matching to an image is provided, try to determine based on annotation name
      xpondingImgs = np.unique(data[REQD_TBL_FIELDS.SRC_IMG_FILENAME].to_numpy())
      # Break into annotaitons by iamge
      for img in xpondingImgs:
        self.addAnnotation(name, data[data[REQD_TBL_FIELDS.SRC_IMG_FILENAME] == img], img)
      return
    image = self._getFullImgName(Path(image))
    # Since only one annotation file can exist per image, concatenate this with any existing files for the same image
    # if needed
    if image.parent != self.imagesDir:
      image = self._copyImgToProj(image)
    annForImg = self.imgToAnnMapping.get(image, None)
    oldAnns = []
    if annForImg is not None and not overwriteOld:
      oldAnns.append(ComponentIO.buildByFileType(annForImg))
    combinedAnns = oldAnns + [data]
    outAnn = pd.concat(combinedAnns, ignore_index=True)
    outAnn[REQD_TBL_FIELDS.INST_ID] = outAnn.index
    outFmt = f".{self.settings['annotation-format']}"
    outName = self.annotationsDir / f'{image.name}{outFmt}'
    ComponentIO.exportByFileType(outAnn, outName, verifyIntegrity=False, readOnly=False, imgDir=self.imagesDir)
    self.imgToAnnMapping[image] = outName

  def _copyImgToProj(self, name: Path, data: NChanImg=None, allowOverwrite=False):
    newName = (self.imagesDir/name.name).resolve()
    if (newName.exists() and not allowOverwrite) or newName == name:
      # Already in the project, no need to copy
      return newName
    if name.exists() and data is None:
      shutil.copy(name, newName)
    elif data is not None:
      # Programmatically created or not from a local file
      # noinspection PyTypeChecker
      io.imsave(newName, data)
    else:
      raise S3AIOError(f'No image data associated with {name.name}. Either the file does not exist or no'
                       f' image information was provided.')
    if name in self.images:
      self.changeImgPath(name, newName)
    return newName

  def _getFullImgName(self, name: Path, thorough=True):
    """
    From an absolute or relative image name, attempts to find the absolute path it corresponds
    to based on current project images. A match is located in the following order:
      - If the image path is already absolute, it is resolved, checked for existence, and returned
      - Solitary project images are searched to see if they end with the specified relative path
      - All base image directories are checked to see if they contain this subpath

    :param thorough: If `False`, as soon as a match is found the function returns. Otherwise,
      all solitary paths and images will be checked to ensure there is exactly one matching
      image for the name provided.
    """
    if name.is_absolute():
      return name.resolve()

    candidates = set()
    strName = str(name)
    for img in self.images:
      if str(img).endswith(strName):
        if not thorough:
          return img
        candidates.add(img)

    for parent in self.baseImgDirs:
      curName = (parent/name).resolve()
      if curName.exists():
        if not thorough:
          return curName
        candidates.add(curName)

    numCandidates = len(candidates)
    if numCandidates != 1:
      msg = f'Exactly one corresponding image file must exist for a given annotation. However,' \
            f' {numCandidates} candidate images were found'
      if numCandidates == 0:
        msg += '.'
      else:
        msg += f':\n{", ".join([c.name for c in candidates])}'
      raise S3AIOError(msg)
    return candidates.pop()

  def exportProj(self, outputFolder: FilePath= 's3a-export'):
    """
    Exports the entire project, making a copy of it at the destination directory
    :param outputFolder:
      helpText: Where to place the exported project
      pType: filepicker
      asFolder: True
    """
    shutil.copytree(self.location, outputFolder)

  @dynamicDocstring(fileTypes=ComponentIO.handledIoTypes_fileFilter().split(';;'))
  def exportAnnotations(self, outputFolder:FilePath= 's3a-export', annotationFormat='csv', combine=False, includeImages=True):
    """
    Exports project annotations, optionally including their source images
    :param outputFolder:
      helpText: Folder for exported annotations
      pType: filepicker
      asFolder: True
    :param annotationFormat:
      helpText: "Annotation file type. E.g. if 'csv', annotations will be saved as csv files. Available
      file types are:
      {fileTypes}"

    :param combine: If `True`, all annotation files will be combined into one exported file with name `annotations.<format>`
    :param includeImages: If `True`, the corresponding image for each annotation will also be exported into an `images`
      folder
    """
    self.saveCfg()
    outputFolder = Path(outputFolder)

    if outputFolder.resolve() == self.cfgFname and not combine:
      return

    outputFolder.mkdir(exist_ok=True)
    if includeImages:
      outImgDir = outputFolder / 'images'
      outImgDir.mkdir(exist_ok=True)
      for img in self.imagesDir.glob('*.*'):
        if self.imgToAnnMapping.get(img, None) is not None:
          shutil.copy(img, outImgDir)

    existingAnnFiles = [f for f in self.imgToAnnMapping.values() if f is not None]
    if combine:
      outAnn = pd.concat(map(ComponentIO.buildByFileType, existingAnnFiles))
      ComponentIO.exportByFileType(outAnn, outputFolder / f'annotations.{annotationFormat}')
    else:
      outAnnsDir = outputFolder / 'annotations'
      if self.settings['annotation-format'] == annotationFormat:
        shutil.copytree(self.annotationsDir, outAnnsDir)
      else:
        for annFile in existingAnnFiles:
          ioArgs = {'imgDir': self.imagesDir}
          ComponentIO.convert(annFile, outAnnsDir/f'{annFile.stem}.{annotationFormat}', ioArgs, ioArgs)


