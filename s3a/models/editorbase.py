import weakref
from functools import wraps
from inspect import isclass
from pathlib import Path
from typing import List, Dict, Any, Union, Collection, Type, Tuple, Sequence, Optional, \
  Callable
from warnings import warn
from enum import Flag, auto

from pyqtgraph.Qt import QtWidgets, QtCore
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem

from s3a.generalutils import frPascalCaseToTitle, frParamToPgParamDict
from s3a.graphicsutils import saveToFile, attemptFileLoad
from s3a.processing import FRAtomicProcess, FRProcessIO, FRGeneralProcWrapper
from s3a.processing.guiwrapper import docParser
from s3a.structures import FRParam, ContainsSharedProps, FilePath, FRParamEditorError, \
  FRS3AWarning

__all__ = ['FRParamEditorBase']

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

def _mkRunBtn(proc: FRAtomicProcess, btnOpts: Union[FRParam, dict]):
  defaultBtnOpts = dict(name=proc.name, type='registeredaction')
  if isinstance(btnOpts, FRParam):
    # Replace falsy helptext with func signature
    if not btnOpts.helpText:
      btnOpts.helpText = docParser(proc.func.__doc__)['top-descr']
    btnOpts = frParamToPgParamDict(btnOpts)
    # Make sure param type is not overridden
    btnOpts.pop('type', None)
  defaultBtnOpts.update(btnOpts)
  if len(proc.input.hyperParamKeys) > 0:
    # In this case, a descriptive name isn't needed since the func name will be
    # present in the parameter group
    defaultBtnOpts['name'] = 'Run'
  runBtn = Parameter.create(**defaultBtnOpts)
  return runBtn

oneOrMultChildren = Union[Sequence[FRParam], FRParam]
_childTuple_asValue = Tuple[FRParam, oneOrMultChildren]
childTuple_asParam = Tuple[FRParam, oneOrMultChildren, bool]
_keyType = Union[_childTuple_asValue, childTuple_asParam]

"""
Eventually, it would be nice to implemenet a global search bar that can find/modify
any action, shortcut, etc. from any parameter. This tracker is an easy way to fascilitate
such a feature. A `class:FRPopupLineEditor` can be created with a model derived from
all parameters from SPAWNED_EDITORS, thereby letting a user see any option from any
param editor.
"""

class FRParamEditorBase(QtWidgets.QDockWidget):
  """
  GUI controls for user-interactive parameters within S3A. Each window consists of
  a parameter tree and basic saving capabilities.
  """
  sigParamStateCreated = Signal(str)
  sigParamStateUpdated = Signal(dict)
  sigParamStateDeleted = Signal(str)

  def __init__(self, parent=None, paramList: List[Dict]=None, saveDir: FilePath='.',
               fileType='param', name=None, topTreeChild: Parameter=None,
               registerCls: Type=None, registerParam: FRParam=None, **registerGroupOpts):
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
    :param registerCls: If this editor was created to hold parameters for a specific
      class, then that class must be provided here. It will ensure registered
      parameters actually appear in this editor. See :func:`FRParamEditor.registerGroup`
    :param registerParam: The grouping parameter to hold registered parameters
      for the regiseterd class. See :func:`FRParamEditor.registerGroup`
    :param registerGroupOpts: These parameters are directly passed as kwargs
      to :func:`FRParamEditor.registerGroup`.
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
        name = frPascalCaseToTitle(name)
      except ValueError:
        name = "Parameter Editor"

    self.groupingToParamMapping: Dict[Any, Optional[FRParam]] = {}
    """
    Allows the editor to associate a class name with its human-readable parameter
    name
    """

    self.registeredFrParams: List[FRParam] = []
    """
    Keeps track of all parameters registerd as properties in this editor. Useful for
    inspecting which parameters are in an editor without traversing the parameter tree
    and reconstructing the name, tooltip, etc.
    """

    self.interactiveProcs: Dict[str, FRAtomicProcess] = {}
    """
    Keeps track of registered functions which have been converted to processes so their
    arguments can be exposed to the user
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
    self.tree = ParameterTree()
    self.tree.setTextElideMode(QtCore.Qt.ElideRight)

    self.params.sigStateChanged.connect(self._paramTreeChanged)

    topParam = self.params
    if topTreeChild is not None:
      topParam = topTreeChild
    self.tree.setParameters(topParam, showTop=False)
    self.tree.header().setSectionResizeMode(QtWidgets.QHeaderView.Interactive)

    # -----------
    # Human readable name (for settings menu)
    # -----------
    self.name: str = name

    # -----------
    # Internal parameters for saving settings
    # -----------
    self.saveDir = Path(saveDir)
    self.fileType = fileType
    self._stateBeforeEdit = self.params.saveState()
    self.lastAppliedName = None

    if registerCls is not None:
      self.registerGroup(registerParam, **registerGroupOpts)(registerCls)
    SPAWNED_EDITORS.append(weakref.proxy(self))

  def _paramTreeChanged(self, rootParam: Parameter, changeDesc: str, data: Tuple[Parameter, int]):
    self._stateBeforeEdit = self.params.saveState()

  # Helper method for accessing simple parameter values
  def __getitem__(self, keys: _keyType):
    """
    Convenience function for accessing child parameters within a parameter editor.
      - If :param:`keys` is a single :class:`FRParam`, the value at that parameter is
        extracted and returned to the user.
      - If :param:`keys` is a :class:`tuple`:

        * The first element of the tuple must correspond to the base name within the
          parameter grouping in order to properly extract the corresponding children.
          For instance, to extract MARGIN from :class:`FRGeneralPropertiesEditor`,
          you must first specify the group parent for that parameter:
            >>> margin = FR_SINGLETON.generalProps[FR_CONSTS.CLS_FOCUSED_IMG_AREA,
            >>>   FR_CONSTS.MARGIN]
        * The second parameter must be a single :class:`FRParam` object or a sequence
          of :class:`FRParam` objects. If a sequence is given, a list of output values
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
    for curKey in keys: # type: FRParam
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
                     allowOverwriteDefault=False, blockWrite=False):
    """
    * Returns dict on successful parameter save and emits sigParamStateCreated.
    * Returns None if no save name was given
    """
    if saveName is None:
      return None
    if paramState is None:
      paramState = self.params.saveState(filter='user')
    # Remove non-useful values
    clearUnwantedParamVals(paramState)
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
    self.setParamTooltips()

  def setParamTooltips(self, expandNameCol=True):
    iterator = QtWidgets.QTreeWidgetItemIterator(self.tree)
    item: QtWidgets.QTreeWidgetItem = iterator.value()
    while item is not None:
      # TODO: Set word wrap on long labels. Currently either can show '...' or wrap but not
      #   both
      # if self.tree.itemWidget(item, 0) is None:
      #   lbl = QtWidgets.QLabel(item.text(0))
      #   self.tree.setItemWidget(item, 0, lbl)
      if (hasattr(item, 'param')
          and 'tip' in item.param.opts
          and len(item.toolTip(0)) == 0
          and self.tree.itemWidget(item, 0) is None):
        item.setToolTip(0, item.param.opts['tip'])
      iterator += 1
      item = iterator.value()
    if expandNameCol:
      self.setAllExpanded(True)

  def setAllExpanded(self, expandedVal=True):
    try:
      topTreeItem: ParameterItem = next(iter(self.params.items))
    except StopIteration:
      return
    for ii in range(topTreeItem.childCount()):
      topTreeItem.child(ii).setExpanded(expandedVal)
    self.tree.resizeColumnToContents(0)

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
    if stateDict is not None:
      return stateDict
    dictFilename = self.formatFileName(stateName)
    stateDict = attemptFileLoad(dictFilename)
    return stateDict

  def deleteParamState(self, stateName: str):
    filename = self.formatFileName(stateName)
    if not filename.exists():
      return
    filename.unlink()
    self.sigParamStateDeleted.emit(stateName)

  def registerProps(self, groupingName: Union[type, str], constParams: List[FRParam],
                    parentParamPath:Collection[str]=None, asProperty=True, **extraOpts):
    """
    Registers a list of proerties and returns an array of each. For parameter descriptions,
    see :func:`FRParamEditor.registerProp`.
    """
    outProps = []
    with self.params.treeChangeBlocker():
      for param in constParams:
        outProps.append(self.registerProp(groupingName, param, parentParamPath,
                                          asProperty, **extraOpts))
    return outProps

  def _addParamGroup(self, groupName: str, paramPath: Sequence[str]=(), **opts):
    paramForCls = Parameter.create(name=groupName, type='group', **opts)
    if len(paramPath) == 0:
      parent = self.params
    else:
      parent = self.params.child(*paramPath)
    parent.addChild(paramForCls)
    return paramForCls

  def registerProp(self, grouping: Union[type, Any], constParam: FRParam,
                   parentParamPath:Collection[str]=None, asProperty=True,
                   objForAssignment=None, **etxraOpts):
    """
    Registers a property defined by *constParam* that will appear in the respective
    parameter editor.

    :param grouping: If *type* it must be a class. In this case, this parameter will
      be listed under the registered class of the same exact name.
      If *str*, *groupingName* must match the exact same string name passed in during
      :func:`registerGroup <FRParamEditor.registerGroup>`. Otherwise, an error will
      occur.
    :param constParam: Object holding parameter attributes such as name, type,
      help text, eREQD_TBL_FIELDS.
    :param parentParamPath: If None, defaults to the top level of the parameters for the
      current class (or paramHolder). *parentParamPath* represents the parent group
      to whom the newly registered parameter should be added
    :param asProperty: If True, creates a property object bound to getter and setter
      for the new param. Otherwise, returns the param itself. If asProperty is false,
      the returned parameter must be evaluated to obtain a value, e.g.
      x = registerProp(..., asProperty=False); myVal = x.value()
    :param etxraOpts: Extra options passed directly to the created :class:`pyqtgraph.Parameter`
    :return: Property bound to this value in the parameter editor
    """
    paramOpts = frParamToPgParamDict(constParam)
    paramOpts.update(etxraOpts)
    paramForEditor = Parameter.create(**paramOpts)

    if grouping not in self.groupingToParamMapping:
      warn(f'The provided grouping "{grouping}" was not recognized, perhaps because '
           f' `registerGroup()` was never called with this grouping. Registering now'
           f' as a top-level grouping.', FRS3AWarning)
      self.registerGroup(None)(grouping)
    groupParam = self.groupingToParamMapping[grouping]

    if groupParam is None:
      paramForCls = self.params
    else:
      paramName = groupParam.name
      paramPath = groupParam.opts.get('parentPath', ()) + (paramName,)
      try:
        paramForCls = self.params.child(*paramPath)
      except KeyError:
        # Parameter wasn't constructed yet
        paramForCls = self._addParamGroup(paramName, paramPath[:-1])

    if parentParamPath is not None and len(parentParamPath) > 0:
      paramForCls = paramForCls.param(*parentParamPath)
    if constParam.name not in paramForCls.names:
      paramForCls.addChild(paramForEditor)

    self.registeredFrParams.append(constParam)
    if not asProperty:
      return paramForEditor

    # Else, create a property and return that instead
    def getAccessorPath():
      parentGrp = self.groupingToParamMapping[grouping]
      accessPath = []
      if parentGrp is not None:
        accessPath.append(parentGrp.name)
      if parentParamPath is not None:
        accessPath.extend(parentParamPath)
      accessPath.append(constParam.name)
      return accessPath

    @property
    def paramAccessor(clsObj):
      accessPath = getAccessorPath()
      return self.params.child(*accessPath).value()

    @paramAccessor.setter
    def paramAccessor(clsObj, newVal):
      accessPath = getAccessorPath()
      param = self.params.child(*accessPath)
      param.setValue(newVal)

    return paramAccessor

  def registerFunc(self, func: Callable, name:str=None, runOpts=RunOpts.BTN,
                   paramPath:Tuple[str,...]=(),
                   btnOpts: Union[FRParam, dict]=None):
    """
    Like `registerProp`, but for functions instead along with interactive parameters
    for each argument. A button is added for the user to force run this function as
    well. In the case of a function with no parameters, the button will be named
    the same as the function itself for simplicity

    :param paramPath:  See `registerProp`
    :param func: Function to make interactive
    :param name: See `FRAtomicProcess.name`
    :param runOpts: Combination of ways this function can be run. Multiple of these
      options can be selected at the same time using the `|` operator.
        * If RunOpts.BTN, a button is present as described.
        * If RunOpts.ON_CHANGE, the function is run when parameter values are
          finished being changed by the user
        * If RunOpts.ON_CHANGING, the function is run every time a value is altered,
          even if the value isn't finished changing.
    :param btnOpts: Overrides defaults for button used to run this function. If
      `RunOpts.BTN` is not in `RunOpts`, these values are ignored.
    """
    proc = FRAtomicProcess(func, name)
    self.interactiveProcs[proc.name] = proc
    # Define caller out here that takes no params so qt signal binding doesn't
    # screw up auto parameter population
    def runProc():
      return proc.run()

    def runpProc_changing(_param: Parameter, newVal: Any):
      forwardedOpts = FRProcessIO(**{_param.name(): newVal})
      return proc.run(forwardedOpts)

    if len(proc.input.hyperParamKeys) > 0:
      topParam = self.params if len(paramPath) == 0 else self.params.child(*paramPath)
      # Check if proc params already exist from a previous addition
      if proc.name not in topParam.names:
        FRGeneralProcWrapper(proc, self, paramPath)
      parentParam = topParam.child(proc.name)
      for param in parentParam:
        if runOpts & RunOpts.ON_CHANGED:
          param.sigValueChanged.connect(runProc)
        if runOpts & RunOpts.ON_CHANGING:
          param.sigValueChanging.connect(runpProc_changing)
    else:
      parentParam = self.params
    if runOpts & RunOpts.BTN:
      runBtn = _mkRunBtn(proc, btnOpts)
      if runBtn.name() in parentParam.names:
        # Bind to existing button intsead
        runBtn = parentParam.child(runBtn.name())
      runBtn.sigActivated.connect(runProc)
      parentParam.addChild(runBtn)
    try:
      self.setParamTooltips(False)
    except AttributeError:
      pass
    return proc

  def registerGroup(self, groupParam: FRParam=None, **opts):
    """
    Intended for use as a class decorator. Registers a class as able to hold
    customizable shortcuts.

    :param groupParam: Parameter holding the name of this class as it should appear
      in this parameter editor. As such, it should be human readable.
      If *None*, params will be shown at the top-level of the parameter tree
    :param opts: Additional registration options. Accepted values:
      :key nameFromParam: This is available so objects without a dedicated class can
        also be registered. In that case, a spoof class is registered under the name
        'groupParam.name' instead of '[decorated class].__qualname__'.
      :key forceCreate: Normally, the class is not registered until at least one property
        (or method) is registered to it. That way, classes registered for other editors
        don't erroneously appear. *forceCreate* will override this rule, creating this
        parameter immediately.
      :key useNewInit: When a class is passed in as a group to register, it is often
        beneficial to ensure its __initEditorParams__ method is called (if it exists) and
        other setup procedures are followed. However, if a param editor is created *inside*
        __initEditorParams__ (or in other circumstances where the class is already initialized
        but an editor is created dynamically), this is an unnecessary and even harmful step.
        Passing *useNewInit*=False will ensure the __init__ method for the registered class
        is *not* altered.
      :key parentPath: If provided, the new group will be created as a child of this
        specified parameter instead of being created as a top level group.
    :return: Undecorated class, but with a new __init__ method which initializes
      all shared properties contained in the '__initEditorParams__' method, if it exists
      in the class. For exceptions to this rule see `key:useNewInit`.
    """
    opts.setdefault('nameFromParam', False)
    def groupingDecorator(grouping: Union[Type, Any]=None):
      if grouping is None or opts['nameFromParam']:
        # In this case nameFromParam must be provided. Use a dummy class for the
        # rest of the proceedings
        grouping = groupParam
      self.groupingToParamMapping.setdefault(grouping, groupParam)

      isAClass = isclass(grouping)


      if isAClass and grouping not in REGISTERED_GROUPINGS:
        REGISTERED_GROUPINGS.add(grouping)
        oldInit = grouping.__init__
        @wraps(oldInit)
        def newInit(clsObj, *args, **kwargs):
          grouping = type(clsObj)
          # groupParam could be inaccurate when initializing base class
          groupParam = self.groupingToParamMapping[grouping]
          if grouping not in INITIALIZED_GROUPINGS and issubclass(grouping, ContainsSharedProps):
            INITIALIZED_GROUPINGS.add(grouping)
            clsObj.__initEditorParams__()
          oldInit(clsObj, *args, **kwargs)
          for editor in SPAWNED_EDITORS:
            try:
              editor._extendedClassInit(clsObj, groupParam)
            except ReferenceError:
              pass
        grouping.__init__ = newInit

      self._extendedGroupingDecorator(grouping, groupParam, **opts)

      parentPath = opts.get('parentPath', ())
      if groupParam is not None:
        groupParam.opts['parentPath'] = parentPath
      if (opts.get('forceCreate', False)
          or len(parentPath) > 0):
        self._addParamGroup(groupParam.name, parentPath)

      return grouping
    if opts['nameFromParam']:
      groupingDecorator()
    return groupingDecorator

  def _extendedClassInit(self, clsObj: Any, groupParam: FRParam):
    """
    For editors that need to perform any initializations within the decorated class,
      they must be able to access the decorated class' *init* function and modify it.
      Allow this by providing an overloadable stub that is inserted into the decorated
      class *init*.
    """
    return

  def _extendedGroupingDecorator(self, cls: Any, groupParam: FRParam, **opts):
    """
    Editors needing additional class decorator boilerplates will place it in this overloaded function
    """


INITIALIZED_GROUPINGS = set()
REGISTERED_GROUPINGS = set()
SPAWNED_EDITORS: List[FRParamEditorBase] = []