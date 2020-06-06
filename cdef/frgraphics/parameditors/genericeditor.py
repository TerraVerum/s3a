import re
from functools import wraps
from pathlib import Path
from typing import List, Dict, Any, Union, Collection, Type, Set

from pyqtgraph.Qt import QtWidgets, QtCore
from pyqtgraph.parametertree import Parameter, ParameterTree

from cdef.frgraphics.graphicsutils import dialogGetSaveFileName, saveToFile, \
  attemptFileLoad
from cdef.structures import FRParam, ContainsSharedProps, FilePath

Signal = QtCore.pyqtSignal

def clearUnwantedParamVals(paramState: dict):
  for k, child in paramState.get('children', {}).items():
    clearUnwantedParamVals(child)
  if paramState.get('value', True) is None:
    paramState.pop('value')


class FRParamEditor(QtWidgets.QDockWidget):
  sigParamStateCreated = Signal(str)
  sigParamStateUpdated = Signal(dict)
  sigParamStateDeleted = Signal(str)

  def __init__(self, parent=None, paramList: List[Dict]=None, saveDir: FilePath='.',
               fileType='param', saveDlgName='Save As', name=None,
               childForOverride: Parameter=None):
    # Place in list so an empty value gets unpacked into super constructor
    if paramList is None:
      paramList = []

    if name is None:
      try:
        propClsName = type(self).__name__
        name = propClsName[:propClsName.index('Editor')]
        name = _frPascalCaseToTitle(name)
      except ValueError:
        name = "Parameter Editor"

    super().__init__(parent)
    self.hide()
    self.setWindowTitle(name)
    self.setObjectName(name)

    self.classNameToParamMapping: Dict[str, FRParam] = {}
    """
    Allows the editor to associate a class name with its human-readable parameter
    name
    """

    self.classInstToEditorMapping: Dict[Any, FRParamEditor] = {}
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
    self._saveDlgName = saveDlgName
    self._stateBeforeEdit = self.params.saveState()
    self.lastAppliedName = None

    # This will be set to 'True' when an action for this editor is added to
    # the main window menu
    self.hasMenuOption = False

    # -----------
    # Additional widget buttons
    # -----------
    self.saveAsBtn = QtWidgets.QPushButton('Save As...')
    self.applyBtn = QtWidgets.QPushButton('Apply')
    self.closeBtn = QtWidgets.QPushButton('Close')

    # -----------
    # Widget layout
    # -----------
    self.dockContentsWidget = QtWidgets.QWidget(parent)
    self.setWidget(self.dockContentsWidget)
    btnLayout = QtWidgets.QHBoxLayout()
    btnLayout.addWidget(self.saveAsBtn)
    btnLayout.addWidget(self.applyBtn)
    btnLayout.addWidget(self.closeBtn)

    self.centralLayout = QtWidgets.QVBoxLayout(self.dockContentsWidget)
    self.centralLayout.addWidget(self.tree)
    self.centralLayout.addLayout(btnLayout)
    # self.setLayout(centralLayout)
    self.tree.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
    # -----------
    # UI Element Signals
    # -----------
    self.saveAsBtn.clicked.connect(self.saveAsBtnClicked)
    self.closeBtn.clicked.connect(self.close)
    self.applyBtn.clicked.connect(self.applyBtnClicked)

  def _paramTreeChanged(self, param, child, idx):
    self._stateBeforeEdit = self.params.saveState()

  def _expandCols(self):
    # totWidth = 0
    for colIdx in range(2):
      self.tree.resizeColumnToContents(colIdx)
    #   totWidth += self.tree.columnWidth(colIdx) + self.tree.margin
    # appInst.processEvents()
    # self.dockContentsWidget.setMinimumWidth(totWidth)
    self.tree.setColumnWidth(0, self.width()//2)
    self.resize(self.tree.width(), self.height())
    self.tree.setMinimumWidth(self.tree.width())


  # Helper method for accessing simple parameter values
  def __getitem__(self, keys: Union[tuple, FRParam, Collection[FRParam]]):
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

    :param keys: One of of the following:
    :return:
    """
    returnSingle = False
    extractObj = False
    if isinstance(keys, tuple):
      if len(keys) > 2:
        extractObj = True
      baseParam = [keys[0].name]
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

  def show(self):
    self.setWindowState(QtCore.Qt.WindowActive)
    # Necessary on MacOS
    self.raise_()
    # Necessary on Windows
    self.activateWindow()
    self.applyBtn.setFocus()
    super().show()


  def reject(self):
    """
    If window is closed apart from pressing 'accept', restore pre-edit state
    """
    self.params.restoreState(self._stateBeforeEdit, removeChildren=False)
    super().reject()

  def applyBtnClicked(self):
    # Don't emit any signals if nothing changed
    newState = self.params.saveState(filter='user')
    outDict = self.params.getValues()
    if self._stateBeforeEdit != newState:
      self._stateBeforeEdit = newState
      self.sigParamStateUpdated.emit(outDict)
    return outDict

  def saveAsBtnClicked(self):
    paramState = self.params.saveState(filter='user')
    saveName = dialogGetSaveFileName(self, self._saveDlgName, self.lastAppliedName)
    self.saveParamState(saveName, paramState)

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
    self.applyBtnClicked()
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
    self.applyBtnClicked()
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

  def registerProp(self, groupingName: Union[type, str], constParam: FRParam,
                   parentParamPath:Collection[str]=None, asProperty=True,
                   objForAssignment=None, **etxraOpts):
    """
    Registers a property defined by *constParam* that will appear in the respective
    parameter editor.

    :param groupingName: If *type* it must be a class. In this case, this parameter will
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

    if isinstance(groupingName, type):
      clsName = groupingName.__qualname__
    else: # This way even if a string wasn't passed in we deal with it like a string
      clsName = str(groupingName)
    paramName = self.classNameToParamMapping[clsName].name
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
      nonlocal clsName
      if not isinstance(groupingName, str):
        clsName = type(clsObj).__qualname__
      if clsObj in self.classInstToEditorMapping:
        xpondingEditor = self.classInstToEditorMapping[clsObj]
      else:
        xpondingEditor = self
      return clsName, xpondingEditor

    # Else, create a property and return that instead
    @property
    def paramAccessor(clsObj):
      trueClsName, xpondingEditor = _paramAccessHelper(clsObj)
      return xpondingEditor[self.classNameToParamMapping[trueClsName], constParam]

    @paramAccessor.setter
    def paramAccessor(clsObj, newVal):
      trueClsName, xpondingEditor = _paramAccessHelper(clsObj)

      param = xpondingEditor[self.classNameToParamMapping[trueClsName], constParam, True]
      param.setValue(newVal)

    return paramAccessor

  def registerGroup(self, groupParam: FRParam, **opts):
    """
    Intended for use as a class decorator. Registers a class as able to hold
    customizable shortcuts.

    :param groupParam: Parameter holding the name of this class as it should appear
      in this parameter editor. As such, it should be human readable.
    :param opts: Additional registration options. Accepted values:
      - nameFromParam: This is available so objects without a dedicated class can
      also be registered. In that case, a spoof class is registered under the name
      'clsParam.name' instead of '[decorated class].__qualname__'.
      - forceCreate: Normally, the class is not registered until at least one property
      (or method) is registered to it. That way, classes registered for other editors
      don't erroneously appear. *forceCreate* will override this rule, creating this
      parameter immediately.
    :return: Undecorated class, but with a new __init__ method which initializes
      all shared properties contained in the '__initEditorParams__' method, if it exists
      in the class.
    """
    opts['nameFromParam'] = opts.get('nameFromParam', False)
    def classDecorator(cls: Union[Type, Any]=None):
      if cls is None:
        # In this case nameFromParam must be provided. Use a dummy class for the
        # rest of the proceedings
        cls = type('DummyClass', (), {})
      if not isinstance(cls, type):
        # Instance was passed, not class
        cls = type(cls)
      if opts['nameFromParam']:
        clsName = groupParam.name
      else:
        clsName = cls.__qualname__
      oldClsInit = cls.__init__
      self._extendedClassDecorator(cls, groupParam, **opts)

      if opts.get('forceCreate', False):
        self._addParamGroup(clsName)

      self.classNameToParamMapping[clsName] = groupParam
      @wraps(oldClsInit)
      def newClassInit(clsObj, *args, **kwargs):
        if (cls not in _INITIALIZED_CLASSES
            and issubclass(cls, ContainsSharedProps)):
          _INITIALIZED_CLASSES.add(cls)
          cls.__initEditorParams__()
          superObj = super(cls, clsObj)
          if isinstance(superObj, ContainsSharedProps):
           superObj.__initEditorParams__()

        if opts.get('saveDefault', True):
          self.saveParamState(saveName='Default', allowOverwriteDefault=True)
        self.classInstToEditorMapping[clsObj] = self
        retVal = oldClsInit(clsObj, *args, **kwargs)
        self._extendedClassInit(clsObj, groupParam)
        return retVal
      cls.__init__ = newClassInit
      return cls
    if opts['nameFromParam']:
      classDecorator()
    return classDecorator

  def _extendedClassInit(self, clsObj: Any, clsParam: FRParam):
    """
    For editors that need to perform any initializations within the decorated class,
      they must be able to access the decorated class' *init* function and modify it.
      Allow this by providing an overloadable stub that is inserted into the decorated
      class *init*.
    """
    return

  def _extendedClassDecorator(self, cls: Any, clsParam: FRParam, **opts):
    """
    Editors needing additional class decorator boilerplates will place it in this overloaded function
    """


_INITIALIZED_CLASSES: Set[Type[ContainsSharedProps]] = set()


def _frPascalCaseToTitle(name: str) -> str:
  """
  Helper utility to turn a FRPascaleCase name to a 'Title Case' title
  :param name: camel-cased name
  :return: Space-separated, properly capitalized version of :param:`Name`
  """
  if not name:
    return name
  if name.startswith('FR'):
    name = name[2:]
  name = re.sub(r'(\w)([A-Z])', r'\1 \2', name)
  return name.title()