from __future__ import annotations

import weakref
from functools import wraps
from inspect import isclass
from pathlib import Path
from typing import List, Dict, Any, Union, Collection, Type, Set, Tuple, Sequence

from pyqtgraph.Qt import QtWidgets, QtCore
from pyqtgraph.parametertree import Parameter, ParameterTree

from s3a.graphicsutils import saveToFile, \
  attemptFileLoad
from s3a.generalutils import frPascalCaseToTitle
from s3a.structures import FRParam, ContainsSharedProps, FilePath, FRParamParseError

__all__ = ['FRParamEditorBase']

Signal = QtCore.Signal

def clearUnwantedParamVals(paramState: dict):
  for k, child in paramState.get('children', {}).items():
    clearUnwantedParamVals(child)
  if paramState.get('value', True) is None:
    paramState.pop('value')

oneOrMultChildren = Union[Sequence[FRParam], FRParam]
_childTuple_asValue = Tuple[FRParam, oneOrMultChildren]
childTuple_asParam = Tuple[FRParam, oneOrMultChildren, bool]
_keyType = Union[_childTuple_asValue, childTuple_asParam]


class FRParamEditorBase(QtWidgets.QDockWidget):
  """
  GUI controls for user-interactive parameters within S3A. Each window consists of
  a parameter tree and basic saving capabilities.
  """
  _spawnedEditors = []
  sigParamStateCreated = Signal(str)
  sigParamStateUpdated = Signal(dict)
  sigParamStateDeleted = Signal(str)

  def __init__(self, parent=None, paramList: List[Dict]=None, saveDir: FilePath='.',
               fileType='param', name=None, childForOverride: Parameter=None,
               registerCls: Type=None, registerParam: FRParam=None):
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
    :param childForOverride: Generally for internal use. If provided, it will
      be inserted into the parameter tree instead of a newly created parameter.
    :param registerCls: If this editor was created to hold parameters for a specific
      class, then that class must be provided here. It will ensure registered
      parameters actually appear in this editor. See :func:`FRParamEditor.registerGroup`
    :param registerParam: The grouping parameter to hold registered parameters
      for the regiseterd class. See :func:`FRParamEditor.registerGroup`
    """
    super().__init__()
    # Place in list so an empty value gets unpacked into super constructor
    if paramList is None:
      paramList = []
    if name is None:
      try:
        propClsName = type(self).__name__
        name = propClsName[:propClsName.index('Editor')]
        name = frPascalCaseToTitle(name)
      except ValueError:
        name = "Parameter Editor"

    self.groupingToParamMapping: Dict[Any, FRParam] = {}
    """
    Allows the editor to associate a class name with its human-readable parameter
    name
    """

    self.classInstToEditorMapping: Dict[Any, FRParamEditorBase] = {}
    """
    For editors that register parameters for *other* editors,
     this allows parameters to be updated from the correct editor
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
    self.params.sigStateChanged.connect(self._paramTreeChanged)
    self.tree = ParameterTree()
    topParam = self.params
    if childForOverride is not None:
      topParam = childForOverride
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
      self.registerGroup(registerParam)(registerCls)
    self._spawnedEditors.append(weakref.proxy(self))

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
        * The second parameter must be a signle :class:`FRParam` objects or a sequence
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
    # Don't emit any signals if nothing changed
    newState = self.params.saveState(filter='user')
    outDict = self.params.getValues()
    if self._stateBeforeEdit != newState:
      self._stateBeforeEdit = newState
      self.sigParamStateUpdated.emit(outDict)
    return outDict

  def saveParamState(self, saveName: str=None, paramState: dict=None,
                     allowOverwriteDefault=False):
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
    self.saveDir.mkdir(parents=True, exist_ok=True)
    saveToFile(paramState, self.formatFileName(saveName),
               allowOverwriteDefault=allowOverwriteDefault)
    self.applyChanges()
    outDict: dict = self.params.getValues()
    self.lastAppliedName = saveName
    self.sigParamStateCreated.emit(saveName)
    return outDict

  def paramDictWithOpts(self, addList: List[str]=None, addTo: List[type(Parameter)]=None,
                        removeList: List[str]=None, paramDict: Dict[str, Any]=None):
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


  def loadParamState(self, stateName: str, stateDict: dict=None,
                     addChildren=False, removeChildren=False):
    loadDict = self._parseStateDict(stateName, stateDict)
    self.params.restoreState(loadDict, addChildren=addChildren, removeChildren=removeChildren)
    self.applyChanges()
    self.lastAppliedName = stateName
    return loadDict

  def formatFileName(self, stateName: str=None):
    if stateName is None:
      stateName = self.lastAppliedName
    return self.saveDir/f'{stateName}.{self.fileType}'

  def _parseStateDict(self, stateName: str, stateDict: dict=None):
    if stateDict is not None:
      return stateDict
    dictFilename = self.formatFileName(stateName)
    stateDict = dict(attemptFileLoad(dictFilename))
    return stateDict

  def deleteParamState(self, stateName: str):
    filename = self.formatFileName(stateName)
    if not filename.exists():
      return
    filename.unlink()
    self.sigParamStateDeleted.emit(stateName)

  def registerProps(self, groupingName: Union[type, str], constParams: List[FRParam],
                    parentParamPath:Collection[str]=None, asProperty=True, **extraOpts):
    outProps = []
    for param in constParams:
      outProps.append(self.registerProp(groupingName, param, parentParamPath,
                                        asProperty, **extraOpts))
    return outProps

  def _addParamGroup(self, groupName: str):
    paramForCls = Parameter.create(name=groupName, type='group')
    paramForCls.sigStateChanged.connect(self._paramTreeChanged)
    self.params.addChild(paramForCls)
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
    :param etxraOpts: Extra options passed directly to the created pyqtgraph.Parameter
    :return: Property bound to this value in the parameter editor
    """
    paramOpts = dict(name=constParam.name, type=constParam.valType, tip=constParam.helpText)
    if constParam.valType == 'group':
      paramOpts.update(children=constParam.value)
    else:
      paramOpts.update(value=constParam.value)
    paramOpts.update(etxraOpts)
    paramForEditor = Parameter.create(**paramOpts)

    groupParam = self.groupingToParamMapping[grouping]
    if groupParam is None:
      paramForCls = self.params
    else:
      paramName = groupParam.name
      if paramName in self.params.names:
        paramForCls = self.params.child(paramName)
      else:
        paramForCls = self._addParamGroup(paramName)

    if parentParamPath is not None and len(parentParamPath) > 0:
      paramForCls = paramForCls.param(*parentParamPath)
    if constParam.name not in paramForCls.names:
      paramForCls.addChild(paramForEditor)

    self._expandCols()

    if not asProperty:
      return paramForEditor

    def _paramAccessHelper(clsObj):
      # when class isn't fully instantiated
      # Retrieve class name from the class instance, since this function call may
      # have resulted from an inhereted class. This only matters if the class name was
      # used instead of a generic string parameter value
      if clsObj in self.classInstToEditorMapping:
        xpondingEditor = self.classInstToEditorMapping[clsObj]
      else:
        xpondingEditor = self
      return xpondingEditor

    # Else, create a property and return that instead
    @property
    def paramAccessor(clsObj):
      xpondingEditor = _paramAccessHelper(clsObj)
      return xpondingEditor[self.groupingToParamMapping[grouping], constParam]

    @paramAccessor.setter
    def paramAccessor(clsObj, newVal):
      xpondingEditor = _paramAccessHelper(clsObj)

      param = xpondingEditor[self.groupingToParamMapping[grouping], constParam, True]
      param.setValue(newVal)

    return paramAccessor

  def registerGroup(self, groupParam: FRParam=None, **opts):
    """
    Intended for use as a class decorator. Registers a class as able to hold
    customizable shortcuts.

    :param groupParam: Parameter holding the name of this class as it should appear
      in this parameter editor. As such, it should be human readable.
      If *None*, params will be shown at the top-level of the parameter tree
    :param opts: Additional registration options. Accepted values:
      - nameFromParam: This is available so objects without a dedicated class can
      also be registered. In that case, a spoof class is registered under the name
      'groupParam.name' instead of '[decorated class].__qualname__'.
      - forceCreate: Normally, the class is not registered until at least one property
      (or method) is registered to it. That way, classes registered for other editors
      don't erroneously appear. *forceCreate* will override this rule, creating this
      parameter immediately.
    :return: Undecorated class, but with a new __init__ method which initializes
      all shared properties contained in the '__initEditorParams__' method, if it exists
      in the class.
    """
    opts.setdefault('nameFromParam', False)
    def groupingDecorator(grouping: Union[Type, Any]=None):
      notAClass = False
      if grouping is None or opts['nameFromParam']:
        # In this case nameFromParam must be provided. Use a dummy class for the
        # rest of the proceedings
        grouping = groupParam
        notAClass = True
      elif not isclass(grouping):
        raise FRParamParseError('Grouping must be either *None* or a class.')

      self._extendedGroupingDecorator(grouping, groupParam, **opts)
      self.groupingToParamMapping.setdefault(grouping, groupParam)

      if opts.get('forceCreate', False):
        self._addParamGroup(groupParam.name)

      if notAClass:
        return
      # Else, need to be sure editor opts are proprely registered on init

      oldClsInit = grouping.__init__

      @wraps(oldClsInit)
      def newClassInit(clsObj, *args, **kwargs):
        # In the case of inheritance, the class type may not match type(clsObj)
        # Redefine grouping here in that event
        cls = type(clsObj)
        groupParam = self.groupingToParamMapping[cls]

        if (cls not in _INITIALIZED_CLASSES
            and issubclass(cls, ContainsSharedProps)):
          _INITIALIZED_CLASSES.add(cls)
          cls.__initEditorParams__()
        # Occurs when parameters are not already initialized from a subclass
        if clsObj not in self.classInstToEditorMapping:
          self.classInstToEditorMapping[clsObj] = self
          doExtendedInit = True
        else:
          doExtendedInit = False

        retVal = oldClsInit(clsObj, *args, **kwargs)

        if doExtendedInit:
          self._extendedClassInit(clsObj, groupParam)
          if opts.get('saveDefault', True):
            self.saveParamState(saveName='Default', allowOverwriteDefault=True)
        return retVal

      grouping.__init__ = newClassInit
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


_INITIALIZED_CLASSES: Set[Type[ContainsSharedProps]] = set()