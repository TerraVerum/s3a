from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, Callable, Any, Dict, List, DefaultDict, Sequence, Union
from warnings import warn

import pandas as pd
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
from s3a.constants import SHORTCUTS_DIR
from s3a.generalutils import helpTextToRichText, getParamChild, clsNameOrGroup
from s3a.models.editorbase import params_flattened
from s3a.structures import PrjParam, ParamEditorError, S3AWarning

from .genericeditor import ParamEditor


def _class_fnNamesFromFnQualname(qualname: str) -> (str, str):
  """
  From the fully qualified function name (e.g. module.class.fn), return the function
  name and class name (module.class, fn).
  :param qualname: output of fn.__qualname__
  :return: (clsName, fnName)
  """
  lastDotIdx = qualname.rfind('.')
  fnName = qualname
  if lastDotIdx < 0:
    # This function isn't inside a class, so defer
    # to the global namespace
    fnParentClass = 'Global'
  else:
    # Get name of class containing this function
    fnParentClass = qualname[:lastDotIdx]
    fnName = qualname[lastDotIdx+1:]
  return fnParentClass, fnName

class EditableShortcut(QtWidgets.QShortcut):
  paramIdx: Tuple[PrjParam, PrjParam]


@dataclass
class BoundFnParams:
  param: PrjParam
  func: Callable
  defaultFnArgs: list


class ShortcutsEditor(ParamEditor):

  def __init__(self, parent=None):

    self.paramToGlobalActsMapping: Dict[PrjParam, QtWidgets.QAction] = {}
    # Unlike other param editors, these children don't get filled in until
    # after the top-level widget is passed to the shortcut editor
    super().__init__(parent, [], saveDir=SHORTCUTS_DIR, fileType='shortcut',
                     name='Tool Shortcuts')

    # Allow global shortcuts
    self.globalParam = PrjParam('Global')

  def _checkUniqueShortcut(self, shortcutOpts: PrjParam):
    # TODO: Find way to preserve old shortcuts, in case multuple other operations
    #   were bound to this shortcut and lost
    if any(shortcutOpts.name == p.name() for p in params_flattened(self.params)):
      self.deleteShortcut(shortcutOpts)

  def _createSeq(self, shortcutOpts: Union[PrjParam, dict],
                 namePath: Sequence[str]=(),
                 overrideBasePath: Sequence[str]=None,
                 **kwargs):
    if overrideBasePath is None:
      namePath = tuple(self._baseRegisterPath) + tuple(namePath)
    else:
      namePath = tuple(overrideBasePath) + tuple(namePath)
    # Round-trip to set helptext, ensure all values are present
    if isinstance(shortcutOpts, dict): shortcutOpts = PrjParam(**shortcutOpts)
    shcForCreate = shortcutOpts.toPgDict()
    shcForCreate['type'] = 'shortcut'
    param = getParamChild(self.params, *namePath, chOpts=shcForCreate)
    param.sigValueChanged.connect(lambda _p, val: self.maybeAmbigWarning(val))
    self.maybeAmbigWarning(param.value())

    return param

  def registerShortcut(self, shortcutOpts: PrjParam, func: Callable,
                   funcArgs: tuple=(), funcKwargs: dict=None, overrideOwnerObj: Any=None,
                   **kwargs):
    self._checkUniqueShortcut(shortcutOpts)
    if funcKwargs is None:
      funcKwargs = {}
    if overrideOwnerObj is None:
      overrideOwnerObj = shortcutOpts.opts.get('ownerObj', None)
    if overrideOwnerObj is None:
      raise ValueError('Solo functions registered to shortcuts must have an owner.\n'
                       f'This is not the case for {func}')
    kwargs.setdefault('namePath', (clsNameOrGroup(overrideOwnerObj),))
    param = self._createSeq(shortcutOpts, **kwargs)
    shc = QtWidgets.QShortcut(overrideOwnerObj)
    shc.setContext(QtCore.Qt.WidgetWithChildrenShortcut)

    shc.activatedAmbiguously.connect(lambda: self.maybeAmbigWarning(param.value()))

    def onChange(_param, key):
      shc.setKey(key)
      self.maybeAmbigWarning(key)
    param.sigValueChanged.connect(onChange)
    onChange(param, param.value())

    def onActivate():
      func(*funcArgs, **funcKwargs)
    shc.activated.connect(onActivate)

    return param

  def registerAction(self, btnOpts: PrjParam, action: QtWidgets.QAction, **kwargs):
    self._checkUniqueShortcut(btnOpts)

    param = self._createSeq(btnOpts, **kwargs)
    action.setToolTip(btnOpts.helpText)
    def shcChanged(_param, newSeq: str):
      action.setShortcut(newSeq)

    param.sigValueChanged.connect(shcChanged)
    shcChanged(None, btnOpts.value)
    return param

  def createRegisteredButton(self, btnOpts: PrjParam,
                             baseBtn: QtWidgets.QAbstractButton=None,
                             asToolBtn=False, **kwargs):
    self._checkUniqueShortcut(btnOpts)
    if asToolBtn:
      btnType = QtWidgets.QToolButton
    else:
      btnType = QtWidgets.QPushButton
    tooltipText = btnOpts.helpText
    if baseBtn is not None:
      newBtn = baseBtn
    elif 'icon' in btnOpts.opts:
      newBtn = btnType(self)
      newBtn.setIcon(QtGui.QIcon(btnOpts.opts['icon']))
      tooltipText = helpTextToRichText(btnOpts.helpText, btnOpts.name)
    else:
      newBtn = btnType(self)
      newBtn.setText(btnOpts.name)
    if btnOpts.value is None:
      # Either the shortcut wasn't given a value or wasn't requested, or already exists
      newBtn.setToolTip(tooltipText)
      return newBtn
    btnOpts.opts['tip'] = tooltipText

    param = self._createSeq(btnOpts, **kwargs)

    def shcChanged(_param, newSeq: str):
      newTooltipText = f'Shortcut: {newSeq}'
      tip = param.opts.get('tip', '')
      tip = helpTextToRichText(tip, newTooltipText)
      newBtn.setToolTip(tip)
    newBtn.setShortcut(param.seqEdit.keySequence())
    param.sigValueChanged.connect(shcChanged)
    shcChanged(None, btnOpts.value)
    return newBtn

  def maybeAmbigWarning(self, shortcut: str):
    conflicts = [p.name() for p in params_flattened(self.params) if p.value() == shortcut]
    if len(conflicts) > 1:
      warn(f'Ambiguous shortcut: {shortcut}\n'
           f'Perhaps multiple shortcuts are assigned the same key sequence? Possible conflicts:\n'
           f'{conflicts}',
           S3AWarning)

  def deleteShortcut(self, shortcutParam: PrjParam):
    matches = [p for p in params_flattened(self.params) if p.name() == shortcutParam['name']]

    formatted = f'<{shortcutParam["name"]}: {shortcutParam["value"]}>'
    if len(matches) == 0:
      warn(f'Shortcut param {formatted} does not exist. No delete performed.')
      return
    for match in matches:
      # Set shortcut key to nothing to prevent its activation
      match.setValue('')
      match.remove()


  def registerProp(self, *args, **etxraOpts):
    """
    Properties should never be registered as shortcuts, so make sure this is disallowed
    """
    raise ParamEditorError('Cannot register property/attribute as a shortcut')

  def registerFunc(self, *args, **kwargs):
    """Functions should not be registered as shortcuts"""
    raise ParamEditorError('Cannot register function as a shortcut. See `registerShortcut`'
                           ' instead.')