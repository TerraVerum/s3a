import weakref
from collections import defaultdict
from contextlib import contextmanager
from enum import Flag, auto
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple, Sequence, Callable, Set, Collection
from warnings import warn

from pyqtgraph.Qt import QtWidgets, QtCore
from pyqtgraph.parametertree import Parameter
from pyqtgraph.parametertree.parameterTypes import GroupParameter

from s3a.generalutils import pascalCaseToTitle, resolveYamlDict, getParamChild
from s3a.graphicsutils import saveToFile, flexibleParamTree, setParamTooltips, \
  expandtreeParams
from s3a.processing import AtomicProcess, ProcessIO, GeneralProcWrapper, ProcessStage, GeneralProcess
from s3a.processing.guiwrapper import docParser
from s3a.structures import PrjParam, FilePath, ParamEditorError, S3AWarning

__all__ = ['ParamEditorBase']

Signal = QtCore.Signal


class RunOpts(Flag):
  NONE = 0
  BTN = auto()
  ON_CHANGED = auto()
  ON_CHANGING = auto()

def clearUnwantedParamVals(paramState: dict):
  for _k, child in paramState.get('children', {}).items():
    clearUnwantedParamVals(child)
  if paramState.get('value', True) is None:
    paramState.pop('value')

def paramState_noDefaults(param: Parameter, extraKeys: Collection[str]=('value',), extraTypes: List[type]=None):
  parentDict = {}
  val = param.value()
  if extraTypes is None:
    extraTypes = set()
  extraTypes = tuple(extraTypes)
  if val != param.defaultValue():
    parentDict['value'] = param.value()
    if not extraTypes or isinstance(param, extraTypes):
      for k in set(extraKeys):
        # False positive
        # noinspection PyUnresolvedReferences
        parentDict[k] = param.opts[k]
  if param.hasChildren() and not parentDict:
    inner = {}
    for child in param:
      chDict = paramState_noDefaults(child)
      if len(chDict) > 0:
        inner[child.name()] = paramState_noDefaults(child)
    if len(inner) > 0:
      parentDict['children'] = inner
  return parentDict


def params_flattened(param: Parameter):
  addList = []
  if 'group' not in param.type():
    addList.append(param)
  for child in param.children(): # type: Parameter
    addList.extend(params_flattened(child))
  return addList

def _mkRunDict(proc: ProcessStage, btnOpts: Union[PrjParam, dict]):
  defaultBtnOpts = dict(name=proc.name, type='registeredaction')
  if isinstance(btnOpts, PrjParam):
    # Replace falsy helptext with func signature
    btnOpts = btnOpts.toPgDict()
  if btnOpts is not None:
    # Make sure param type is not overridden
    btnOpts.pop('type', None)
    defaultBtnOpts.update(btnOpts)
  if len(defaultBtnOpts.get('tip', '')) == 0 and isinstance(proc, AtomicProcess):
    defaultBtnOpts['tip'] = docParser(proc.func.__doc__)['top-descr']
  if len(proc.input.hyperParamKeys) > 0:
    # In this case, a descriptive name isn't needed since the func name will be
    # present in the parameter group
    defaultBtnOpts['name'] = 'Run'
  return defaultBtnOpts

oneOrMultChildren = Union[Sequence[PrjParam], PrjParam]
_childTuple_asValue = Tuple[PrjParam, oneOrMultChildren]
childTuple_asParam = Tuple[PrjParam, oneOrMultChildren, bool]
_keyType = Union[_childTuple_asValue, childTuple_asParam]

"""
Eventually, it would be nice to implemenet a global search bar that can find/modify
any action, shortcut, etc. from any parameter. This tracker is an easy way to fascilitate
such a feature. A `class:FRPopupLineEditor` can be created with a model derived from
all parameters from SPAWNED_EDITORS, thereby letting a user see any option from any
param editor.
"""

class ParamEditorBase(QtWidgets.QDockWidget):
  """
  GUI controls for user-interactive parameters within S3A. Each window consists of
  a parameter tree and basic saving capabilities.
  """
  sigParamStateCreated = Signal(str)
  sigParamStateUpdated = Signal(dict)
  sigParamStateDeleted = Signal(str)

  _baseRegisterPath: Sequence[str] = ()
  """
  Classes typically register all their properites in bulk under the same group of
  parameters. This property will be overridden (see :meth:`setBaseRegisterPath`) by
  the class name of whatever class is currently registering properties.
  """

  def __init__(self, parent=None, paramList: List[Dict] = None, saveDir: FilePath = '.',
               fileType='param', name=None, topTreeChild: Parameter = None,
               **kwargs):
    """
    GUI controls for user-interactive parameters within S3A. Each window consists of
    a parameter tree and basic saving capabilities.

    :param parent: GUI parent of this window
    :param paramList: User-editable parameters. This is often *None* and parameters
      are added dynamically within the code.
    :param saveDir: When "save" is performed, the resulting settings will be saved
      here.
    :param fileType: The filetype of the saved settings. E.g. if a settings configuration
      is saved with the name "test", it will result in a file "test.&lt;fileType&gt;"
    :param name: User-readable name of this parameter editor
    :param topTreeChild: Generally for internal use. If provided, it will
      be inserted into the parameter tree instead of a newly created parameter.
    """
    super().__init__(parent)
    cls = type(self)
    # Place in list so an empty value gets unpacked into super constructor
    if paramList is None:
      paramList = []
    if name is None:
      try:
        propClsName = cls.__name__
        name = propClsName[:propClsName.index('Editor')]
        name = pascalCaseToTitle(name)
      except ValueError:
        name = "Parameter Editor"

    self.registeredPrjParams: List[PrjParam] = []
    """
    Keeps track of all parameters registerd as properties in this editor. Useful for
    inspecting which parameters are in an editor without traversing the parameter tree
    and reconstructing the name, tooltip, etc.
    """

    self.procToParamsMapping: Dict[ProcessStage, GroupParameter] = {}
    """
    Keeps track of registered functions (or prcesses) and their associated
    gui parameters
    """

    self.instantiatedClassTypes = set()
    """
    Records whether classes with registered parameters have been instantiated. This way,
    base classes with registered parameters but no instances will not appear in the
    parameter editor.
    """

    # -----------
    # Construct parameter tree
    # -----------
    self.params = Parameter.create(name='Parameters', type='group', children=paramList)
    self.tree = flexibleParamTree()

    self.params.sigStateChanged.connect(self._paramTreeChanged)

    topParam = self.params
    if topTreeChild is not None:
      topParam = topTreeChild
    self.tree.setParameters(topParam, showTop=False)

    # -----------
    # Human readable name (for settings menu)
    # -----------
    self.name: str = name

    # -----------
    # Internal parameters for saving settings
    # -----------
    if saveDir is not None:
      saveDir = Path(saveDir)
    self.saveDir = saveDir
    self.fileType = fileType
    self._stateBeforeEdit = self.params.saveState()
    self.lastAppliedName = None

    self.setAllExpanded = lambda expandedVal=True: expandtreeParams(self.tree, expandedVal)
    self.setParamTooltips = lambda expandNameCol=True: setParamTooltips(self.tree, expandNameCol)

    SPAWNED_EDITORS.append(weakref.proxy(self))

  def _paramTreeChanged(self, rootParam: Parameter, changeDesc: str, data: Tuple[Parameter, int]):
    self._stateBeforeEdit = self.params.saveState()

  # Helper method for accessing simple parameter values
  def __getitem__(self, keys: _keyType):
    """
    Convenience function for accessing child parameters within a parameter editor.
      - If :param:`keys` is a single :class:`PrjParam`, the value at that parameter is
        extracted and returned to the user.
      - If :param:`keys` is a :class:`tuple`:

        * The first element of the tuple must correspond to the base name within the
          parameter grouping in order to properly extract the corresponding children.
          For instance, to extract MARGIN from :class:`GeneralPropertiesEditor`,
          you must first specify the group parent for that parameter:
            >>> margin = FR_SINGLETON.generalProps[PRJ_CONSTS.CLS_FOCUSED_IMG_AREA,
            >>>   PRJ_CONSTS.MARGIN]
        * The second parameter must be a single :class:`PrjParam` object or a sequence
          of :class:`PrjParam` objects. If a sequence is given, a list of output values
          respecting input order is provided.
        * The third parameter is optional. If provided, the :class:`Parameter<pyqtgraph.Parameter>`
          object is returned instead of the :func:`value()<Parameter.value>` data
          *within* the object.

    :param keys: As explained above.
    :return: Either a :class:`Parameter<pyqtgraph.Parameter>` or value of that parameter,
    """
    returnSingle = False
    extractObj = False
    if isinstance(keys, tuple):
      if len(keys) > 2:
        extractObj = True
      baseParam = [keys[0].name] if keys[0] is not None else []
      keys = keys[1]
    else:
      baseParam = []
    if not hasattr(keys, '__iter__'):
      keys = [keys]
      returnSingle = True
    outVals = []
    # Account for the case where the child params are all top-level
    if baseParam is None: baseParam = ()
    extractFunc = lambda name: self.params.child(*baseParam, name)
    if not extractObj:
      oldExtractFunc = extractFunc
      extractFunc = lambda name: oldExtractFunc(name).value()
    for curKey in keys: # type: PrjParam
      outVals.append(extractFunc(curKey.name))
    if returnSingle:
      return outVals[0]
    else:
      return outVals

  def applyChanges(self):
    """Broadcasts that this parameter editor has updated changes"""
    # Don't emit any signals if nothing changed
    newState = self.params.saveState(filter='user')
    outDict = self.params.getValues()
    if self._stateBeforeEdit != newState:
      self._stateBeforeEdit = newState
      self.sigParamStateUpdated.emit(outDict)
    return outDict

  def saveParamState(self, saveName: str=None, paramState: dict=None,
                     allowOverwriteDefault=False, blockWrite=False,
                     saveChangedOnly=True):
    """
    * Returns dict on successful parameter save and emits sigParamStateCreated.
    * Returns None if no save name was given
    """
    if saveName is None or self.saveDir is None:
      return None
    if paramState is None and allowOverwriteDefault:
      paramState = self.params.saveState(filter='user')
      clearUnwantedParamVals(paramState)
    elif paramState is None:
      paramState = paramState_noDefaults(self.params)
    # Remove non-useful values
    if not blockWrite and self.saveDir is not None:
      self.saveDir.mkdir(parents=True, exist_ok=True)
      saveToFile(paramState, self.formatFileName(saveName),
                 allowOverwriteDefault=allowOverwriteDefault)
    # self.applyChanges()
    self.lastAppliedName = saveName
    self.sigParamStateCreated.emit(str(saveName))
    return paramState

  def saveCurStateAsDefault(self):
    self.saveParamState('Default', allowOverwriteDefault=True)
    self.setParamTooltips(self.tree)

  def paramDictWithOpts(self, addList: List[str]=None, addTo: List[type(Parameter)]=None,
                        removeList: List[str]=None, paramDict: Dict[str, Any]=None):
    """
    Allows customized alterations to which portions of a pyqtgraph parameter will be saved
    in the export. The default option only allows saving all or no extra options. This
    allows you to specify which options should be saved, and what parameter types they
    should be saved for.

    :param addList: Options to include in the export for *addTo* type parameters
    :param addTo: Which parameter types should get these options
    :param removeList: Options to exclude in the export for *addTo* type parameters
    :param paramDict: The initial export that should be modified. This is usually the
      output of `Parameter().saveState(filter='user')`
    :return: Modified version of :paramDict: with alterations as explained above
    """
    if addList is None:
      addList = []
    if addTo is None:
      addTo = []
    if removeList is None:
      removeList = []
    def addCustomOpts(dictRoot, paramRoot: Parameter):
      for pChild in paramRoot:
        dChild = dictRoot['children'][pChild.name()]
        addCustomOpts(dChild, pChild)
      if type(paramRoot) in addTo:
        for opt in addList:
          if opt in paramRoot.opts:
            dictRoot[opt] = paramRoot.opts[opt]
      for opt in removeList:
        if dictRoot.get(opt, True) is None:
          dictRoot.pop(opt)
    if paramDict is None:
      paramDict = self.params.saveState('user')
    addCustomOpts(paramDict, self.params)
    return paramDict

  def loadParamState(self, stateName: Union[str, Path], stateDict: dict=None,
                     addChildren=False, removeChildren=False, applyChanges=True):
    loadDict = self._parseStateDict(stateName, stateDict)
    self.params.restoreState(loadDict, addChildren=addChildren, removeChildren=removeChildren)
    if applyChanges:
      self.applyChanges()
    self.lastAppliedName = stateName
    return loadDict

  def loadFromPartialNames(self, stateName: Union[str, Path],
                           stateDict: dict=None,
                           candidateParams: List[Parameter]=None,
                           applyChanges=True):
    loadDict = self._parseStateDict(stateName, stateDict)
    if candidateParams is None:
      candidateParams = params_flattened(self.params)
    def titleOrName(param):
      return param.opts['title'] or param.name()
    for kk, vv in loadDict.items():
      if isinstance(vv, dict):
        # Successively traverse down child tree
        candidateParams = [p for p in candidateParams if titleOrName(p.parent()) == kk]
        self.loadFromPartialNames('', vv, candidateParams)
        del loadDict[kk]
    with self.params.treeChangeBlocker():
      for kk, vv in loadDict.items():
        matches = [p for p in candidateParams if titleOrName(p) == kk]
        if len(matches) == 1:
          matches[0].setValue(vv)
        elif len(matches) == 0:
          warn(f'No matching parameters for key {kk}. Ignoring.', S3AWarning)
        else:
          raise ParamEditorError(f'Multiple matching parameters for key {kk}:\n'
                                 f'{matches}')
    if applyChanges:
      self.applyChanges()


  def formatFileName(self, stateName: Union[str, Path]=None):
    stateName = Path(stateName)
    if stateName is None:
      stateName = self.lastAppliedName
    if stateName.is_absolute():
      statePathPlusStem = stateName
    else:
      statePathPlusStem = self.saveDir/stateName
    return statePathPlusStem.with_suffix(f'.{self.fileType}')

  def _parseStateDict(self, stateName: Union[str, Path], stateDict: dict=None):
    return resolveYamlDict(self.formatFileName(stateName), stateDict)[1]

  def deleteParamState(self, stateName: str):
    filename = self.formatFileName(stateName)
    if not filename.exists():
      return
    filename.unlink()
    self.sigParamStateDeleted.emit(stateName)

  def registerProps(self, constParams: List[PrjParam], namePath:Sequence[str]=(),
                     asProperty=True, **extraOpts):
    """
    Registers a list of proerties and returns an array of each. For parameter descriptions,
    see :func:`PrjParamEditor.registerProp`.
    """
    outProps = []
    with self.params.treeChangeBlocker():
      for param in constParams:
        outProps.append(self.registerProp(param, namePath, asProperty, **extraOpts))
    return outProps

  def registerProp(self, constParam: PrjParam=None, namePath: Sequence[str]=(),
                   asProperty=True, overrideBasePath: Sequence[str]=None, **etxraOpts):
    """
    Registers a property defined by *constParam* that will appear in the respective
    parameter editor.

    :param constParam: Object holding parameter attributes such as name, type,
      help text, etc. If *None*, defaults to a 'group' type
    :param namePath: If None, defaults to the top level of the parameters for the
      current class (or paramHolder). *namePath* represents the parent group
      to whom the newly registered parameter should be added
    :param asProperty: If True, creates a property object bound to getter and setter
      for the new param. Otherwise, returns the param itself. If asProperty is false,
      the returned parameter must be evaluated to obtain a value, e.g.
      x = registerProp(..., asProperty=False); myVal = x.value()
    :param overrideBasePath: Whether to use the base path specified by ParamEditor._baseRegisterPath
      (if *None*) or this specified override
    :param etxraOpts: Extra options passed directly to the created :class:`pyqtgraph.Parameter`

    :return: Property bound to this value in the parameter editor
    """
    paramOpts = constParam.toPgDict()
    paramOpts.update(etxraOpts)
    paramForEditor = Parameter.create(**paramOpts)
    if overrideBasePath is None:
      namePath = tuple(self._baseRegisterPath) + tuple(namePath)
    else:
      namePath = tuple(overrideBasePath) + tuple(namePath)
    paramForCls = getParamChild(self.params, *namePath)

    if constParam.name not in paramForCls.names:
      paramForCls.addChild(paramForEditor)

    self.registeredPrjParams.append(constParam)
    if not asProperty:
      return paramForEditor

    @property
    def paramAccessor(clsObj):
      return self.params.child(*namePath, constParam.name).value()

    @paramAccessor.setter
    def paramAccessor(clsObj, newVal):
      param = self.params.child(*namePath, constParam.name)
      param.setValue(newVal)

    return paramAccessor

  def registerFunc(self, func: Callable, *, runOpts=RunOpts.BTN,
                   namePath:Tuple[str, ...]=(),
                   paramFormat = pascalCaseToTitle,
                   overrideBasePath: Sequence[str]=None,
                   btnOpts: Union[PrjParam, dict]=None,
                   nest=True,
                   returnParam=False,
                   **kwargs):
    """
    Like `registerProp`, but for functions instead along with interactive parameters
    for each argument. A button is added for the user to force run this function as
    well. In the case of a function with no parameters, the button will be named
    the same as the function itself for simplicity

    :param namePath:  See `registerProp`
    :param func: Function to make interactive
    :param runOpts: Combination of ways this function can be run. Multiple of these
      options can be selected at the same time using the `|` operator.
        * If RunOpts.BTN, a button is present as described.
        * If RunOpts.ON_CHANGE, the function is run when parameter values are
          finished being changed by the user
        * If RunOpts.ON_CHANGING, the function is run every time a value is altered,
          even if the value isn't finished changing.
    :param paramFormat: Formatter which turns variable names into display names. The default takes variables in pascal
      case (e.g. variableName) or snake case (e.g. variable_name) and converts to Title Case (e.g. Variable Name).
      Custom functions must have the signature (str) -> str.
    :param overrideBasePath: See :meth:`~ParamEditor.registerProp`
    :param btnOpts: Overrides defaults for button used to run this function. If
      `RunOpts.BTN` is not in `RunOpts`, these values are ignored.
    :param nest: If *True*, functions with multiple default arguments will have these nested
      inside a group parameter bearing the function name. Otherwise, they will be added
      directly to the parent parameter specified by `namePath` + `baseRegisterPath`
    :param returnParam: Whether to return the parent parameter associated with this newly
      registered function
    :param kwargs: All additional kwargs are passed to AtomicProcess when wrapping the function.
    """
    if not isinstance(func, ProcessStage):
      proc: ProcessStage = AtomicProcess(func, **kwargs)
    else:
      proc = func
    # Define caller out here that takes no params so qt signal binding doesn't
    # screw up auto parameter population
    def runProc():
      return proc.run()

    def runpProc_changing(_param: Parameter, newVal: Any):
      forwardedOpts = ProcessIO(**{_param.name(): newVal})
      return proc.run(forwardedOpts)

    if overrideBasePath is None:
      namePath = tuple(self._baseRegisterPath) + tuple(namePath)
    else:
      namePath = tuple(overrideBasePath) + tuple(namePath)

    topParam = getParamChild(self.params, *namePath)
    if len(proc.input.hyperParamKeys) > 0:
      # Check if proc params already exist from a previous addition
      wrapped = GeneralProcWrapper(proc, topParam, paramFormat, treatAsAtomic=True, nestHyperparams=nest)
      parentParam = wrapped.parentParam
      for param in parentParam:
        if runOpts & RunOpts.ON_CHANGED:
          param.sigValueChanged.connect(runProc)
        if runOpts & RunOpts.ON_CHANGING:
          param.sigValueChanging.connect(runpProc_changing)
    else:
      parentParam: GroupParameter = topParam
    if runOpts & RunOpts.BTN:
      runBtnDict = _mkRunDict(proc, btnOpts)
      if not nest:
        # Make sure button name is correct
        runBtnDict['name'] = proc.name
      runBtn = getParamChild(parentParam, chOpts=runBtnDict)
      runBtn.sigActivated.connect(runProc)
    try:
      self.setParamTooltips(False)
    except AttributeError:
      pass
    self.procToParamsMapping[proc] = parentParam

    if returnParam:
      return proc, parentParam
    return proc

  @classmethod
  @contextmanager
  def setBaseRegisterPath(cls, *path: str):
    oldPath = cls._baseRegisterPath
    cls._baseRegisterPath = path
    yield
    cls._baseRegisterPath = oldPath

SPAWNED_EDITORS: List[ParamEditorBase] = []