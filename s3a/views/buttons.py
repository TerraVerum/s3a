from __future__ import annotations

from typing import Collection, Callable, Union, Iterable, Any, Dict, Sequence, List, Optional

from pyqtgraph.Qt import QtWidgets
from pyqtgraph.parametertree import Parameter
from utilitys import fns, PrjParam, ParamEditor

from s3a import parameditors

__all__ = ['ButtonCollection']

class _DEFAULT_OWNER: pass
"""None is a valid owner, so create a sentinel that's not valid"""
btnCallable = Callable[[PrjParam], Any]
class ButtonCollection(QtWidgets.QGroupBox):
  def __init__(self, parent=None, title: str=None, btnParams: Collection[PrjParam]=(),
               btnTriggerFns: Union[btnCallable, Collection[btnCallable]]=(),
               exclusive=True, asToolBtn=True,
               **createOpts):
    super().__init__(parent)
    self.lastTriggered: Optional[PrjParam] = None
    self.uiLayout = QtWidgets.QHBoxLayout(self)
    self.btnGroup = QtWidgets.QButtonGroup(self)
    self.paramToFuncMapping: Dict[PrjParam, btnCallable] = dict()
    self.paramToBtnMapping: Dict[PrjParam, QtWidgets.QPushButton] = dict()
    self.asToolBtn = asToolBtn
    if title is not None:
      self.setTitle(title)
    self.btnGroup.setExclusive(exclusive)

    if not isinstance(btnTriggerFns, Iterable):
      btnTriggerFns = [btnTriggerFns]*len(btnParams)
    for param, fn in zip(btnParams, btnTriggerFns):
      self.create_addBtn(param, fn, **createOpts)

  def create_addBtn(self, btnParam: PrjParam, triggerFn: btnCallable, checkable=False, **registerOpts):
    if btnParam in self.paramToBtnMapping:
      # Either already exists or wasn't designed to be a button
      return
    registerOpts.setdefault('asToolBtn', self.asToolBtn)
    newBtn = parameditors.PRJ_SINGLETON.shortcuts.createRegisteredButton(btnParam, **registerOpts)
    if checkable:
      newBtn.setCheckable(True)
      oldTriggerFn = triggerFn
      # If the button is checkable, only call this function when the button is checked
      def newTriggerFn(param: PrjParam):
        if newBtn.isChecked():
          oldTriggerFn(param)
      triggerFn = newTriggerFn
    newBtn.clicked.connect(lambda: self.callFuncByParam(btnParam))

    self.addBtn(btnParam, newBtn, triggerFn)

  def clear(self):
    for button in self.paramToBtnMapping.values():
      self.btnGroup.removeButton(button)
      self.uiLayout.removeWidget(button)
      button.deleteLater()

    self.paramToBtnMapping.clear()
    self.paramToFuncMapping.clear()

  def addFromExisting(self, other: ButtonCollection, which: Collection[PrjParam]=None):
    for (param, btn), func in zip(other.paramToBtnMapping.items(), other.paramToFuncMapping.values()):
      if which is None or param in which:
        self.addBtn(param, btn, func)

  def addBtn(self, param: PrjParam, btn: QtWidgets.QPushButton, func: btnCallable):
    self.btnGroup.addButton(btn)
    self.uiLayout.addWidget(btn)
    self.paramToFuncMapping[param] = func
    self.paramToBtnMapping[param] = btn


  def callFuncByParam(self, param: PrjParam):
    if param is None:
      return
    # Ensure function is called in the event it requires a button to be checked
    btn = self.paramToBtnMapping[param]
    if btn.isCheckable():
      btn.setChecked(True)
    self.paramToFuncMapping[param](param)
    self.lastTriggered = param

  def addByParam(self, param: Parameter, copy=True, **registerOpts):
    """
    Adds a button to a group based on the parameter. Also works for group params
    that have an acttion nested.
    """
    for param in fns.params_flattened(param):
      curCopy = copy
      if param.type() in ['action', 'shortcut'] and param.opts.get('guibtn', True):
        existingBtn = None
        try:
          existingBtn = next(iter(param.items)).button
        except (StopIteration, AttributeError):
          curCopy = True
        if curCopy:
          self.create_addBtn(PrjParam(**param.opts), lambda *args: param.activate(), **registerOpts)
        else:
          self.addBtn(PrjParam(**param.opts), existingBtn, existingBtn.click)

  @classmethod
  def fromToolsEditors(cls,
                       editors: Sequence[ParamEditor],
                       title='Tools',
                       ownerClctn: ButtonCollection=None,
                       **registerOpts):
    if ownerClctn is None:
      ownerClctn = ButtonCollection(title=title, exclusive=True)

    for editor in editors:
      ownerClctn.addByParam(editor.params, **registerOpts)

    return ownerClctn

  def toolbarFormat(self):
    """
    Returns a list of buttons + title in a format that's easier to add to a toolbar, e.g.
    doesn't require as much horizontal space
    """
    title = self.title()
    out: List[QtWidgets.QWidget] = [] if title is None else [QtWidgets.QLabel(self.title())]
    for btn in self.paramToBtnMapping.values():
      out.append(btn)
    return out