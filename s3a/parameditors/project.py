import shutil
from pathlib import Path
from typing import List

from skimage import io
from typing_extensions import TypedDict

from s3a.constants import BASE_DIR
from s3a.generalutils import resolveYamlDict
from s3a.graphicsutils import saveToFile
from s3a.parameditors.table import TableData
from s3a.processing import AtomicProcess
from s3a.structures import FilePath, NChanImg
from s3a.parameditors.genericeditor import ParamEditor


class _ImageDict(TypedDict):
  name: str
  data: NChanImg

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
    self.images: List[_ImageDict] = []
    self.annotations = []
    self.exportOpts = {}

  def loadCfg(self, cfgFname: FilePath=None, cfgDict: dict = None):
    _, defaultCfg = resolveYamlDict(BASE_DIR/'projectcfg.yml')
    cfgFname, cfgDict = resolveYamlDict(cfgFname, cfgDict)
    hierarchicalUpdate(defaultCfg, cfgDict)
    cfg = self.cfg = defaultCfg
    tableInfo = cfg.get('table-cfg', {})
    if isinstance(tableInfo, str):
      tableDict = None
      tableName = tableInfo
    else:
      tableDict = tableInfo
      tableName = cfgFname
    self.tableData.loadCfg(tableName, tableDict)

    for imgInfo in cfg.get('images', []):
      curImg: _ImageDict = {}
      if isinstance(imgInfo, str):
        curImg['name'] = imgInfo
        curImg['data'] = io.imread(imgInfo)
      elif isinstance(imgInfo, dict):
        curImg = imgInfo
      self.images.append(curImg)

class ProjectEditor(ParamEditor):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.projectData = ProjectData()
    self.createProject = AtomicProcess(self.createProject, ignoreKeys=['projCfg'])

  def createProject(self, *, location: FilePath= '.', projName: FilePath=None, projCfg: dict=None):
    """
    Creates a new project with the specified settings in the specified directory.
    :param location:
      helpText: Where to create the project
      pType: filePicker
    :param projName:
      title: Project Name
      helpText: see `ProjectData.loadCfg` for information
      pType: string
    :param projCfg: see `ProjectData.loadCfg` for information
    """
    location = Path(location)
    location.mkdir(exist_ok=True, parents=True)
    annotationDir = location/'annotations'
    annotationDir.mkdir(exist_ok=True)

    data = self.projectData
    if projName is None:
      projName = Path(location/'projectcfg.yml')
      projCfg = {}
    data.loadCfg(projName, projCfg)

    if data.cfg['import-opts']['copy-data']:
      imagesDir = location / 'images'
      imagesDir.mkdir(exist_ok=True)
      for img in data.images: # type: _ImageDict
        newName = imagesDir/Path(img['name']).name
        io.imsave(newName, img['data'])
        img['name'] = str(newName.relative_to(location))

      newAnns = []
      for ann in data.annotations:
        newName = annotationDir/Path(ann).name
        shutil.copy(ann, newName)
        newAnns.append(str(newName))
      data.annotations.clear()
      data.annotations.extend(newAnns)

      newName = data.tableData.cfgFname.name
      saveToFile(data.tableData.cfg, location/newName, True)
      data.tableData.cfgFname = newName
      data.cfg['table-cfg'] = newName

    projName = Path(projName).name
    saveToFile(data.cfg, location/projName)