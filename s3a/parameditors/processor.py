from __future__ import annotations

from inspect import isclass
from pathlib import Path
from typing import Optional, Dict, List, Callable, Union

from imageprocessing.processing import ImageProcess
from pyqtgraph.Qt import QtCore
from pyqtgraph.parametertree import Parameter
from pyqtgraph.parametertree.parameterTypes import ListParameter

from s3a.constants import MENU_OPTS_DIR
from s3a.structures import FRParam, \
  FRAlgProcessorError, FRParamEditorError
from s3a.views.procwrapper import FRImgProcWrapper
from .genericeditor import FRParamEditor
from .pgregistered import FRProcGroupParameter
from s3a.generalutils import frPascalCaseToTitle

Signal = QtCore.Signal

class FRAlgPropsMgr(FRParamEditor):
  # sigProcessorCreated = Signal(object) # Signal(FRAlgCollectionEditor)
  def __init__(self, parent=None):
    super().__init__(parent, fileType='', saveDir='')
    self.processorCtors : List[Callable[[], ImageProcess]] = []
    self.spawnedCollections : List[FRAlgCollectionEditor] = []

  def registerGroup(self, groupParam: FRParam = None, **opts):
    raise FRParamEditorError("Individual processors shouldn't be registered as groups."
                             " They should be spawned from an AlgCollectionEditor.")

  def createProcessorForClass(self, clsObj, editorName=None) -> FRAlgCollectionEditor:
    if not isclass(clsObj):
      clsObj = type(clsObj)
    clsName = clsObj.__name__
    formattedClsName = frPascalCaseToTitle(clsName)
    editorDir = MENU_OPTS_DIR/formattedClsName.lower()
    if editorName is None:
      # Strip "FR" from class name before retrieving name
      editorName = formattedClsName + ' Processor'
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
                     topTreeChild=self.algOpts)
    self.expandAllBtn.hide()
    self.collapseAllBtn.hide()

    self.saveDir.mkdir(parents=True, exist_ok=True)

    self.curProcessor: Optional[FRImgProcWrapper] = None
    self.nameToProcMapping: Dict[str, FRImgProcWrapper] = {}

    self.VERT_LST_NAMES = ['fgVerts', 'bgVerts']

    wrapped : Optional[FRImgProcWrapper] = None
    for processorCtor in procCtors:
      # Retrieve proc so default can be set after
      wrapped = self.addImageProcessor(processorCtor())
    self.algOpts.setDefault(wrapped)
    self.switchActiveProcessor(proc=wrapped)
    # self.saveParamState('Default', allowOverwriteDefault=True)

  def addImageProcessor(self, newProc: ImageProcess):
    processor = FRImgProcWrapper(newProc, self)
    self.tree.addParameters(self.params.child(processor.algName))

    self.nameToProcMapping.update({processor.algName: processor})
    self.algOpts.setLimits(self.nameToProcMapping.copy())
    return processor

  def saveParamState(self, saveName: str=None, paramState: dict=None,
                     allowOverwriteDefault=False, blockWrite=False):
    """
    The algorithm editor also needs to store information about the selected algorithm, so lump
    this in with the other parameter information before calling default save.
    """
    if paramState is None:
      paramDict = self.paramDictWithOpts(addList=['enabled'], addTo=[FRProcGroupParameter],
                                         removeList=['value'])
      paramState = {'Selected Algorithm': self.algOpts.value().algName,
                    'Parameters': paramDict}
    return super().saveParamState(saveName, paramState, allowOverwriteDefault, blockWrite)

  def loadParamState(self, stateName: Union[str, Path], stateDict: dict=None,
                     addChildren=False, removeChildren=False, applyChanges=True):
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