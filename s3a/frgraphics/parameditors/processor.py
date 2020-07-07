from __future__ import annotations

from inspect import isclass
from os.path import join
from typing import Optional, Dict, List, Callable, Union

from imageprocessing.processing import ImageProcess
from pyqtgraph.Qt import QtCore
from pyqtgraph.parametertree import Parameter
from pyqtgraph.parametertree.parameterTypes import ListParameter

from s3a.procwrapper import FRImgProcWrapper
from s3a.projectvars import MENU_OPTS_DIR
from s3a.structures import FRComplexVertices, FRParam, \
  FRAlgProcessorError
from .genericeditor import FRParamEditor
from .pgregistered import FRProcGroupParameter
from ...generalutils import frPascalCaseToTitle

Signal = QtCore.Signal

class FRAlgPropsMgr(FRParamEditor):
  # sigProcessorCreated = Signal(object) # Signal(FRAlgCollectionEditor)
  def __init__(self, parent=None):
    super().__init__(parent, fileType='', saveDir='')
    self.processorCtors : List[Callable[[], ImageProcess]] = []
    self.spawnedCollections : List[FRAlgCollectionEditor] = []

  def registerGroup(self, groupParam: FRParam, **opts):
    # Don't save a default file for this class
    return super().registerGroup(groupParam, saveDefault=False, **opts)

  def createProcessorForClass(self, clsObj, editorName=None) -> FRAlgCollectionEditor:
    if not isclass(clsObj):
      clsObj = type(clsObj)
    clsName = clsObj.__name__
    editorDir = join(MENU_OPTS_DIR, clsName, '')
    if editorName is None:
      # Strip "FR" from class name before retrieving name
      editorName = frPascalCaseToTitle(clsName) + ' Processor'
    newEditor = FRAlgCollectionEditor(editorDir, self.processorCtors, name=editorName)
    self.spawnedCollections.append(newEditor)
    # Wrap in property so changes propagate to the calling class
    lims = newEditor.algOpts.opts['limits']
    defaultKey = next(iter(lims))
    defaultAlg = lims[defaultKey]
    newEditor.algOpts.setDefault(defaultAlg)
    newEditor.switchActiveProcessor(proc=defaultAlg)
    # self.sigProcessorCreated.emit(newEditor)
    return newEditor

  def addProcessCtor(self, procCtor: Callable[[], ImageProcess]):
    self.processorCtors.append(procCtor)
    for algCollection in self.spawnedCollections:
      algCollection.addImageProcessor(procCtor())

class FRAlgCollectionEditor(FRParamEditor):
  def __init__(self, saveDir, procCtors: List[Callable[[], ImageProcess]],
               name=None, parent=None):
    algOptDict = {
      'name': 'Algorithm', 'type':  'list', 'values': [], 'value': 'N/A'
    }
    self.treeAlgOpts: Parameter = Parameter(name='Algorithm Selection', type='group', children=[algOptDict])
    self.algOpts: ListParameter = self.treeAlgOpts.children()[0]
    self.algOpts.sigValueChanged.connect(lambda param, proc: self.switchActiveProcessor(proc))
    super().__init__(parent, saveDir=saveDir, fileType='alg', name=name,
                     childForOverride=self.algOpts)

    self.saveDir.mkdir(parents=True, exist_ok=True)

    self.curProcessor: Optional[FRImgProcWrapper] = None
    self.nameToProcMapping: Dict[str, FRImgProcWrapper] = {}

    self.VERT_LST_NAMES = ['fgVerts', 'bgVerts']
    self.vertBuffers: Dict[str, FRComplexVertices] = {
      vType: FRComplexVertices() for vType in self.VERT_LST_NAMES
    }

    wrapped : Optional[FRImgProcWrapper] = None
    for processorCtor in procCtors:
      # Retrieve proc so default can be set after
      wrapped = self.addImageProcessor(processorCtor())
    self.algOpts.setDefault(wrapped)
    self.switchActiveProcessor(proc=wrapped)
    self.saveParamState('Default', allowOverwriteDefault=True)

  def addImageProcessor(self, newProc: ImageProcess):
    processor = FRImgProcWrapper(newProc, self)
    self.tree.addParameters(self.params.child(processor.algName))

    self.nameToProcMapping.update({processor.algName: processor})
    self.algOpts.setLimits(self.nameToProcMapping.copy())
    return processor

  def saveParamState(self, saveName: str=None, paramState: dict=None,
                     allowOverwriteDefault=False):
    """
    The algorithm editor also needs to store information about the selected algorithm, so lump
    this in with the other parameter information before calling default save.
    """
    paramDict = self.paramDictWithOpts(addList=['enabled'], addTo=[FRProcGroupParameter],
                                       removeList=['value'])
    def addEnabledOpt(dictRoot, paramRoot: Parameter, prevRoot=None):
      for pChild in paramRoot:
        dChild = dictRoot['children'][pChild.name()]
        addEnabledOpt(dChild, pChild)
      if isinstance(paramRoot, FRProcGroupParameter):
        dictRoot['enabled'] = paramRoot.opts['enabled']
    paramState = {'Selected Algorithm': self.algOpts.value().algName,
                  'Parameters': paramDict}
    return super().saveParamState(saveName, paramState, allowOverwriteDefault)

  def loadParamState(self, stateName: str, stateDict: dict=None, addChildren=False, removeChildren=False):
    stateDict = self._parseStateDict(stateName, stateDict)
    selectedOpt = stateDict.get('Selected Algorithm', None)
    # Get the impl associated with this option name
    isLegitSelection = selectedOpt in self.algOpts.opts['limits']
    if not isLegitSelection:
      selectedImpl = self.algOpts.value()
      raise FRAlgProcessorError(f'Selection {selectedOpt} does'
                                f' not match the list of available algorithms. Defaulting to {selectedImpl}')
    else:
      selectedImpl = self.algOpts.opts['limits'][selectedOpt]
    self.algOpts.setValue(selectedImpl)
    super().loadParamState(stateName, stateDict['Parameters'])

  def switchActiveProcessor(self, proc: Union[str, FRImgProcWrapper]):
    """
    Changes which processor is active. if FRImgProcWrapper, uses that as the processor.
    If str, looks for that name in current processors and uses that
    """
    if isinstance(proc, str):
      proc = self.nameToProcMapping[proc]
    # Hide all except current selection
    # TODO: Find out why hide() isn't working. Documentation indicates it should
    # Instead, use the parentChanged utility as a hacky workaround
    selectedParam = self.params.child(proc.algName)
    for ii, child in enumerate(self.params.children()):
      shouldHide = child is not selectedParam
      # Offset by 1 to account for self.algOpts
      self.tree.setRowHidden(1 + ii, QtCore.QModelIndex(), shouldHide)
    # selectedParam.show()
    self.curProcessor = proc