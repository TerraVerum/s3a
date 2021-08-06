import argparse
import inspect
import os.path
from contextlib import ExitStack
from pathlib import Path
from typing import Optional, Union, Type, Dict, List, Sequence
from warnings import warn

import numpy as np
import pandas as pd
from pandas import DataFrame as df
from pyqtgraph.Qt import QtCore, QtWidgets

from utilitys import EditorPropsMixin, RunOpts, ParamEditorPlugin, fns, ParamContainer, \
  DeferredActionStackMixin as DASM, ParamEditorDockGrouping
from .. import ComponentIO, defaultIo
from ..constants import PRJ_CONSTS, REQD_TBL_FIELDS, PRJ_ENUMS
from ..controls.tableviewproxy import CompDisplayFilter, CompSortFilter
from ..logger import getAppLogger
from ..models.tablemodel import ComponentMgr
from ..parameditors.appstate import AppStateEditor
from ..plugins import INTERNAL_PLUGINS, tablefield, EXTERNAL_PLUGINS
from ..plugins.file import FilePlugin
from ..plugins.misc import RandomToolsPlugin
from ..shared import SharedAppSettings
from ..structures import FilePath, NChanImg
from ..views.imageareas import MainImage
from ..views.tableview import CompTableView

__all__ = ['S3ABase']

class S3ABase(DASM, EditorPropsMixin, QtWidgets.QMainWindow):
  """
  Top-level widget for producing component bounding regions from an input image.
  """

  sigRegionAccepted = QtCore.Signal()
  __groupingName__ = 'S3A Window'

  scope = ExitStack()
  """
  Allows each instance of s3a to act like a "scope" for all objecats instantiated within. Keeps multiple instances
  of separate S3A pieces from e.g. sharing the same undo buffer. This is managed by __new__.
  """

  sharedAttrs: SharedAppSettings
  """App-level properties that many moving pieces use"""

  def __new__(cls, *args, **kwargs):
    cls.scope.close()
    cls.scope, newAttrs = cls.createScope(cls.scope, returnAttrs=True)
    newAttrs: SharedAppSettings
    obj = super().__new__(cls, *args, **kwargs)
    obj.sharedAttrs = newAttrs
    return obj

  def __initEditorParams__(self, shared: SharedAppSettings):
    self.props = ParamContainer()
    with shared.generalProps.setBaseRegisterPath(PRJ_CONSTS.CLS_COMP_EXPORTER.name):
      shared.generalProps.registerProps(
          [PRJ_CONSTS.EXP_ONLY_VISIBLE, PRJ_CONSTS.INCLUDE_FNAME_PATH],
        container=self.props)

  @staticmethod
  def createScope(scope: ExitStack = None, returnAttrs=False):
    if scope is None:
      scope = ExitStack()
    newAttrs = SharedAppSettings()
    scope.enter_context(EditorPropsMixin.setOpts(shared=newAttrs))
    scope.enter_context(DASM.setStack(newAttrs.actionStack))
    if returnAttrs:
      return scope, newAttrs
    return scope

  def __init__(self, parent=None, **startupSettings):
    super().__init__(parent)

    self.clsToPluginMapping: Dict[Type[ParamEditorPlugin], ParamEditorPlugin] = {}
    """
    Maintains a record of all plugins added to this window. Only up to one instance of each plugin class is expected.
    """

    self.docks: List[QtWidgets.QDockWidget] = []
    """List of docks from added plugins"""

    self.tblFieldToolbar = QtWidgets.QToolBar('Table Field Plugins')
    self.generalToolbar = QtWidgets.QToolBar('General')

    self.mainImg = MainImage(toolbar=self.generalToolbar)
    PRJ_CONSTS.TOOL_ACCEPT_FOC_REGION.opts['ownerObj'] = self.mainImg
    attrs = self.sharedAttrs
    self.mainImg.toolsEditor.registerFunc(self.acceptFocusedRegion,
                                          btnOpts=PRJ_CONSTS.TOOL_ACCEPT_FOC_REGION)
    _, param = attrs.generalProps.registerFunc(attrs.actionStack.resizeStack,
                                                       name=self.__groupingName__,
                                                       runOpts=RunOpts.ON_CHANGED,
                                                       maxLength=PRJ_CONSTS.PROP_UNDO_BUF_SZ.value,
                                                       returnParam=True)
    param.child('maxLength').setOpts(title=PRJ_CONSTS.PROP_UNDO_BUF_SZ.name)
    self.statBar = QtWidgets.QStatusBar(self)
    self.menuBar_ = self.menuBar()

    attrs.shortcuts.registerShortcut(PRJ_CONSTS.TOOL_CLEAR_ROI,
                                             self.mainImg.clearCurRoi,
                                             overrideOwnerObj=self.mainImg
                                             )

    self.compMgr = ComponentMgr()
    # Register exporter to allow user parameters
    ComponentIO.tableData = attrs.tableData

    self.compTbl = CompTableView()
    self.compDisplay = CompDisplayFilter(self.compMgr, self.mainImg, self.compTbl)

    self.compTbl.setSortingEnabled(True)
    self.compTbl.setAlternatingRowColors(True)
    # Allow filtering/sorting
    self.sortFilterProxy = CompSortFilter(self.compMgr)
    self.compTbl.setModel(self.sortFilterProxy)

    self.hasUnsavedChanges = False
    self.srcImgFname: Optional[Path] = None

    self.appStateEditor = AppStateEditor(self, name='App State Editor')

    # -----
    # INTERFACE WITH QUICK LOADER / PLUGINS
    # -----
    toAdd = INTERNAL_PLUGINS() + EXTERNAL_PLUGINS()
    # Insert "settings" and "shortcuts" in a more logical location (after file + edit)
    toAdd = toAdd[:2] + [self.sharedAttrs.settingsPlg, self.sharedAttrs.shortcutsPlg] \
              + toAdd[2:]
    for plg in toAdd:
      if inspect.isclass(plg):
        self.addPlugin(plg)
      else:
        self._addPluginObj(plg)

    # Create links for commonly used plugins
    # noinspection PyTypeChecker
    self.filePlg: FilePlugin = self.clsToPluginMapping[FilePlugin]
    attrs.tableData.sigCfgUpdated.connect(lambda: self.resetTblFields())

    # noinspection PyTypeChecker
    self.vertsPlg: tablefield.VerticesPlugin = self.clsToPluginMapping[tablefield.VerticesPlugin]
    # noinspection PyTypeChecker
    self.miscPlugin: RandomToolsPlugin = self.clsToPluginMapping[RandomToolsPlugin]
    self.compIo = self.filePlg.projData.compIo

    # Connect signals
    # -----
    # COMPONENT MANAGER
    # -----
    def handleUpdate(*_args):
      self.hasUnsavedChanges = True
    self.compMgr.sigCompsChanged.connect(handleUpdate)

    # -----
    # MAIN IMAGE
    # -----
    def handleCompsChanged(changedDict: dict):
      ser = self.mainImg.compSer
      focusedId = ser[REQD_TBL_FIELDS.INST_ID]
      if focusedId in changedDict['deleted']:
        self.changeFocusedComp()
      elif focusedId in changedDict['changed']:
        ser.update(self.compMgr.compDf.loc[focusedId])
    self.compMgr.sigCompsChanged.connect(handleCompsChanged)

    # -----
    # COMPONENT TABLE
    # -----
    self.compDisplay.sigCompsSelected.connect(lambda newComps: self.changeFocusedComp(newComps.index))

    # -----
    # MISC
    # -----
    self.saveAllEditorDefaults()

  def resetTblFields(self):
    """
    When table fields change, the displayed columns must change and the view
    must be made aware. Ensure this occurs here
    """
    # Even if the field names are the same, e.g. classes may added or default values could
    # be changed. So, reset the cell editor delegates no matter what
    # Start by adding any potentially new plugins
    for plg in self.filePlg.projData.spawnedPlugins:
      self._addPluginObj(plg)
    self.compTbl.setColDelegates()
    self.compTbl.popup.reflectDelegateChange()
    # Make sure this is necessary, first
    for mgr in self.compMgr, self.compTbl.popup.tbl.mgr:
      if mgr.colTitles == list([f.name for f in self.sharedAttrs.tableData.allFields]):
        # Fields haven't changed since last reset. Types could be different, but nothing
        # will break. So, the table doesn't have to be completely reset
        return

      mgr.beginResetModel()
      mgr.rmComps()
      mgr.resetFields()
      mgr.endResetModel()

  def saveAllEditorDefaults(self):
    for editor in self.docks:
      if isinstance(editor, ParamEditorDockGrouping):
        for subEditor in editor.editors:
          subEditor.saveCurStateAsDefault()
      else:
        editor.saveCurStateAsDefault()

  @DASM.undoable('Accept Focused Region')
  def acceptFocusedRegion(self):
    """Applies the focused image vertices to the corresponding component in the table"""
    # If the component was deleted
    mgr = self.compMgr
    focusedId = self.mainImg.compSer[REQD_TBL_FIELDS.INST_ID]
    exists = focusedId in mgr.compDf.index
    if not exists and focusedId != REQD_TBL_FIELDS.INST_ID.value:
      # Could be a brand new component, allow in that case
      warn('Cannot accept region as this component was deleted.', UserWarning)
      return

    self.sigRegionAccepted.emit()

    ser = self.mainImg.compSer
    if ser[REQD_TBL_FIELDS.VERTICES].isEmpty():
      # Component should be erased. Since new components will not match existing
      # IDs the same function will work regardless of whether this was new or existing
      self.compMgr.rmComps([ser[REQD_TBL_FIELDS.INST_ID]])
      return

    if exists:
      undo = self._acceptFocused_existing(ser)
    else:
      undo = self._acceptFocused_new(ser)
    self.changeFocusedComp()
    yield
    undo()

  def _acceptFocused_new(self, compSer: pd.Series):
    # New, make a brand new table entry
    compAsDf = fns.serAsFrame(compSer)
    newIds = self.compMgr.addComps(compAsDf)['added']
    compAsDf[REQD_TBL_FIELDS.INST_ID] = newIds
    compAsDf = compAsDf.set_index(REQD_TBL_FIELDS.INST_ID, drop=False)
    def undo():
      self.compMgr.rmComps(newIds)
      # Make sure the old, previously existing outline re-exists at this point
      self.vertsPlg.updateRegionFromDf(compAsDf)
    return undo

  def _acceptFocused_existing(self, compSer: pd.Series):
    oldComp = self.compMgr.compDf.loc[[compSer[REQD_TBL_FIELDS.INST_ID]]].copy()
    modifiedDf = fns.serAsFrame(compSer)
    self.compMgr.addComps(modifiedDf, addtype=PRJ_ENUMS.COMP_ADD_AS_MERGE)
    def undo():
      self.add_focusComps(oldComp, addType=PRJ_ENUMS.COMP_ADD_AS_MERGE)
      self.compDisplay.regionPlot.focusById(modifiedDf.index)
      self.mainImg.updateFocusedComp(compSer)
    return undo

  def clearBoundaries(self):
    """Removes all components from the component table"""
    self.compMgr.rmComps()

  def addPlugin(self, pluginCls: Type[ParamEditorPlugin], *args, **kwargs):
    """
    From a class inheriting the *PrjParamEditorPlugin*, creates a plugin object
    that will appear in the S3A toolbar. An entry is created with dropdown options
    for each editor in *pluginCls*'s *editors* attribute.

    :param pluginCls: Class containing plugin actions
    :param args: Passed to class constructor
    :param kwargs: Passed to class constructor
    """
    if pluginCls in self.clsToPluginMapping:
      getAppLogger(__name__).info(f'Ignoring {pluginCls} since it was previously added', UserWarning)

    plugin: ParamEditorPlugin = pluginCls(*args, **kwargs)
    return self._addPluginObj(plugin)

  def _addPluginObj(self, plugin: ParamEditorPlugin, overwriteExisting=False):
    """
    Adds already intsantiated plugin. Discourage public use of this API since most plugin use should be class-based
    until window registration. This mainly provides for adding spawned plugins from prject data
    """
    pluginCls = type(plugin)
    if not overwriteExisting and pluginCls in self.clsToPluginMapping:
      return None
    self.clsToPluginMapping[pluginCls] = plugin
    if plugin.dock is not None and plugin.dock not in self.docks:
      self.docks.append(plugin.dock)
    plugin.attachWinRef(self)
    if plugin.dock:
      plugin.dock.setParent(self)
    return plugin

  @DASM.undoable('Change Main Image')
  def setMainImg(self, fileName: FilePath=None, imgData: NChanImg=None,
                 clearExistingComps=True):
    """
    * If fileName is None, the main and focused images are blacked out.
    * If only fileName is provided, it is assumed to be an image. The image data
    will be populated by reading in that file.
    * If both fileName and imgData are provided, then imgData is used to populate the
    image, and fileName is assumed to be the file associated with that data.

    :param fileName: Filename either to load or that corresponds to imgData
    :param imgData: N-Channel numpy image
    :param clearExistingComps: If True, erases all existing components on image load.
      Else, they are retained.
    """
    oldFile = self.srcImgFname
    oldData = self.mainImg.image
    if fileName is not None:
      fileName = Path(fileName).resolve()
    if fileName == self.srcImgFname:
      return

    self.saveCurAnnotation()

    if imgData is not None:
      self.mainImg.setImage(imgData)
    else:
      self.mainImg.setImage(fileName)
    self.srcImgFname = fileName

    self.clearBoundaries()
    self.mainImg.plotItem.vb.autoRange()
    if fileName is not None:
      # Add image data if the file doesn't exist
      data = None if fileName.exists() else self.mainImg.image
      self.filePlg.projData.addImage(fileName, data)
    self.loadNewAnnotations()
    infoName = (fileName and fileName.name) or None
    getAppLogger(__name__).info(f'Changed main image to {infoName}')
    yield
    self.setMainImg(oldFile, oldData, clearExistingComps)

  def saveCurAnnotation(self):
    srcImg = self.srcImgFname
    if srcImg is None:
      return
    srcImg_proj = self.filePlg.imagesDir/srcImg.name
    if not srcImg_proj.exists() or srcImg_proj != srcImg:
      # Either the image didn't exist (i.e. was programmatically generated) or doesn't yet belong to the project
      self.filePlg.addImage(srcImg, data=self.mainImg.image, copyToProj=True,
                            allowOverwrite=True)
    # srcImg_proj is guaranteed to exist at this point
    self.filePlg.addAnnotation(data=self.exportableDf, image=srcImg_proj, overwriteOld=True)
    self.srcImgFname = srcImg_proj
    self.hasUnsavedChanges = False
    getAppLogger(__name__).info('Saved current annotation')

  def loadNewAnnotations(self, imgFname: FilePath=None):
    if imgFname is None:
      imgFname = self.srcImgFname
    if imgFname is None:
      return
    imgAnns = self.filePlg.imgToAnnMapping.get(imgFname, None)
    if imgAnns is not None:
      self.compMgr.addComps(self.compIo.importByFileType(imgAnns, imShape=self.mainImg.image.shape))
      # 'hasUnsavedChanges' will be true after this, even though the changes are saved.
      self.hasUnsavedChanges = False

  @fns.dynamicDocstring(filters=defaultIo.ioFileFilter(PRJ_ENUMS.IO_EXPORT))
  def exportCurAnnotation(self, outFname: Union[str, Path], **kwargs):
    """
    Exports current image annotations to a file. This may be more convenient than exporting
    an entire project if just the current current annotations are needed

    :param outFname:
      title: Output File
      helpText: Where to export. The file extension determines the save type
      pType: filepicker
      existing: False
      fileFilter: {filters}
    :param kwargs: Passed to the exporter
    """
    outFname = Path(outFname)
    self.compIo.exportByFileType(self.exportableDf, outFname, imShape=self.mainImg.image.shape,
                                 **kwargs)
    msgPath = os.path.join(outFname.parent.name, outFname.name)
    getAppLogger(__name__).attention(f'Exported current annotation to {msgPath}')
  @property
  def exportableDf(self):
    """
    Dataframe from manager with populated information for main image name and
    potentially filtered to only visible components (if requested by the user)
    """
    displayIds = self.compDisplay.displayedIds
    srcImgFname = self.srcImgFname
    if self.props[PRJ_CONSTS.EXP_ONLY_VISIBLE] and displayIds is not None:
      exportIds = displayIds
    else:
      exportIds = self.compMgr.compDf.index
    exportDf: df = self.compMgr.compDf.loc[exportIds].copy()
    if not self.props[PRJ_CONSTS.INCLUDE_FNAME_PATH] and srcImgFname is not None:
      # Only use the file name, not the whole path
      srcImgFname = srcImgFname.name
    elif srcImgFname is not None:
      srcImgFname = str(srcImgFname)
    # Assign correct export name for only new components
    overwriteIdxs = exportDf[REQD_TBL_FIELDS.SRC_IMG_FILENAME] == PRJ_CONSTS.ANN_CUR_FILE_INDICATOR.value
    # TODO: Maybe the current filename will match the current file indicator. What happens then?
    exportDf.loc[overwriteIdxs, REQD_TBL_FIELDS.SRC_IMG_FILENAME] = srcImgFname
    return exportDf

  def openAnnotations(self, inFname: str, loadType=PRJ_ENUMS.COMP_ADD_AS_NEW):
    pathFname = Path(inFname)
    if self.mainImg.image is None:
      raise IOError('Cannot load components when no main image is set.')
    fType = pathFname.suffix[1:]
    if not any(fType in typ for typ in self.compIo.importTypes):
      raise IOError(f'Extension {fType} is not recognized. Must be one of:\n'
                    + self.compIo.ioFileFilter())
    newComps = self.compIo.importByFileType(inFname, self.mainImg.image.shape)
    self.compMgr.addComps(newComps, loadType)

  @DASM.undoable('Create New Component', asGroup=True)
  def add_focusComps(self, newComps: df, addType=PRJ_ENUMS.COMP_ADD_AS_NEW):
    changeDict = self.compMgr.addComps(newComps, addType)
    # Focus is performed by comp table
    # Arbitrarily choose the last possible component
    changeList = np.concatenate([changeDict['added'], changeDict['changed']])
    if len(changeList) == 0:
      return changeDict
    self.changeFocusedComp(changeList[-1])
    return changeDict

  def changeFocusedComp(self, compIds: Union[int, Sequence[int]]=None):
    # TODO: More robust scenario if multiple comps are in the dataframe
    #   For now, treat ambiguity by not focusing anything
    if np.isscalar(compIds):
      compIds = [compIds]
    if compIds is None or len(compIds) != 1 or compIds[0] not in self.compMgr.compDf.index:
      self.compDisplay.regionPlot.focusById([])
      self.mainImg.updateFocusedComp()
    else:
      newComp: pd.Series = self.compMgr.compDf.loc[compIds[0]]
      newCompId = newComp[REQD_TBL_FIELDS.INST_ID]
      self.compDisplay.regionPlot.focusById([newCompId])
      self.mainImg.updateFocusedComp(newComp)

  # Stolen and adapted for python from https://stackoverflow.com/a/42910109/9463643
  # noinspection PyTypeChecker
  def addTabbedDock(self, area: QtCore.Qt.DockWidgetArea, dockwidget: QtWidgets.QDockWidget):
    curAreaWidgets = [d for d in self.findChildren(QtWidgets.QDockWidget)
                      if self.dockWidgetArea(d) == area]
    try:
      self.tabifyDockWidget(curAreaWidgets[-1], dockwidget)
    except IndexError:
      # First dock in area
      self.addDockWidget(area, dockwidget)

  def makeHelpOpts(self, parser: argparse.ArgumentParser=None):
    """Adds quick loader and app state options to a parser, or creates a new parser if one is not passed"""
    if parser is None:
      parser = argparse.ArgumentParser('S3A')
    ql = self.appStateEditor.quickLoader
    for editor in ql.listModel.uniqueEditors:
      states = ql.listModel.getParamStateFiles(editor.saveDir, editor.fileType)
      formatted = [f'"{s}"' for s in states]
      parser.add_argument(f'--{editor.name.lower().replace(" ", "")}',
                          choices=formatted)
    for loader in self.appStateEditor.stateFuncsDf.index:
      parser.add_argument(f'--{loader}', type=str)