from s3a.generalutils import resolveYamlDict
from s3a.parameditors import TableData
from s3a.structures import FilePath
from skimage import io

from .genericeditor import ParamEditor


class ProjectData:
  def __init__(self):
    self.tableData = TableData()
    self.cfg = {}
    self.images = []
    self.annotations = []
    self.exportOpts = {}

  def loadCfg(self, cfgFname: FilePath=None, cfgDict: dict = None):
    cfgFname, cfgDict = resolveYamlDict(cfgFname, cfgDict)
    tableInfo = cfgDict.get('table-cfg', {})
    if isinstance(tableInfo, str):
      tableDict = None
      tableName = tableInfo
    else:
      tableDict = tableInfo
      tableName = 'projectcfg.yml'
    self.tableData.loadCfg(tableName, tableDict)

    for imgInfo in cfgDict.get('images', []):
      curImg = {}
      if isinstance(imgInfo, str):
        curImg['name'] = imgInfo
        curImg['data'] = io.imread(imgInfo)
      elif isinstance(imgInfo, dict):
        curImg = imgInfo
      self.images.append(curImg)

    for annInfo in cfgDict.get('annotations', []):
      self.annotations.append(annInfo)

    self.exportOpts = cfgDict.get('export-opts', {})
    self.exportOpts.setdefault('annotation-format', 'csv')
    self.exportOpts.setdefault('include-images', True)
    self.exportOpts.setdefault('combine-annotations', False)


class ProjectEditor(ParamEditor):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)