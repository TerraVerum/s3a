from functools import partial
from typing import Optional, List

from pyqtgraph.Qt import QtGui, QtWidgets, QtCore
from pyqtgraph.parametertree import parameterTypes, Parameter
from pyqtgraph.parametertree.parameterTypes import ActionParameterItem, ActionParameter

from ..graphicsutils import findMainWin


class FRShortcutParameterItem(parameterTypes.WidgetParameterItem):
  """
  Class for creating custom shortcuts. Must be made here since pyqtgraph doesn't
  provide an implementation.
  """

  def __init__(self, param, depth):
    super().__init__(param, depth)
    self.item: Optional[QtGui.QKeySequence] = None

  def makeWidget(self):
    item = QtWidgets.QKeySequenceEdit()

    item.sigChanged = item.editingFinished
    item.value = lambda: item.keySequence().toString()
    item.setValue = item.setKeySequence
    self.item = item
    return self.item

  def updateDisplayLabel(self, value=None):
    # Make sure the key sequence is human readable
    self.displayLabel.setText(self.widget.keySequence().toString())

  # def contextMenuEvent(self, ev: QtGui.QContextMenuEvent):
  #   menu = self.contextMenu
  #   delAct = QtWidgets.QAction('Set Blank')
  #   delAct.triggered.connect(lambda: self.widget.setValue(''))
  #   menu.addAction(delAct)
  #   menu.exec(ev.globalPos())


class FRShortcutParameter(Parameter):
  itemClass = FRShortcutParameterItem

  def __init__(self, **opts):
    super().__init__(**opts)

class FRActionWithShortcutParameterItem(ActionParameterItem):
  def __init__(self, param: Parameter, depth):
    super().__init__(param, depth)
    shortcutSeq = param.opts.get('shortcutSeq', '')

    # shcLabel = QtWidgets.QLabel('Shortcut: ', self.layoutWidget)
    # self.layout.addWidget(shcLabel)

    self.keySeqEdit = QtWidgets.QKeySequenceEdit(shortcutSeq)
    # Without the main window as a parent, the shortcut will not activate when
    # the quickloader is hidden
    # TODO: Maybe it is desirable for shortcuts to only work when quickloader
    self.shortcut = QtWidgets.QShortcut(shortcutSeq, findMainWin())
    self.shortcut.activated.connect(self.buttonClicked)
    button: QtWidgets.QPushButton = self.button
    tip = self.param.opts.get('tip', None)
    if tip is not None:
      button.setToolTip(tip)
    def updateShortcut(newSeq: QtGui.QKeySequence):
      self.shortcut.setKey(newSeq)
      param.opts['shortcutSeq'] = newSeq.toString()
    self.keySeqEdit.keySequenceChanged.connect(updateShortcut)

    # Make sure that when a parameter is removed, the shortcut is also deactivated
    param.sigRemoved.connect(lambda: self.shortcut.setParent(None))

    self.layout.addWidget(self.keySeqEdit)

class FRActionWithShortcutParameter(ActionParameter):
  itemClass = FRActionWithShortcutParameterItem

class FRCustomMenuParameter(parameterTypes.GroupParameter):
  def __init__(self, menuActions: List[str]=None, **opts):
    if menuActions is None:
      menuActions: List[str] = []
    self.menuActions = menuActions
    self.item = None
    super().__init__(**opts)

  def makeTreeItem(self, depth):
    item = super().makeTreeItem(depth)
    self.item = item
    if not hasattr(item, 'contextMenu'):
      item.contextMenu = QtWidgets.QMenu()
    item.contextMenuEvent = lambda ev: item.contextMenu.popup(ev.globalPos())
    for actName in self.menuActions:
      act = item.contextMenu.addAction(actName)
      act.triggered.connect(partial(self.menuActTriggered, actName))
    return item

  def menuActTriggered(self, actName: str):
    # Toggle 'enable' on click
    return

  def setOpts(self, **opts):
    super().setOpts()

_toggleName = 'Toggle Enable'
class FRProcGroupParameter(FRCustomMenuParameter):
  def __init__(self, **opts):
    menuActions = opts.pop('menuActions', [])
    if _toggleName not in menuActions:
      menuActions.append(_toggleName)
    super().__init__(menuActions=menuActions, **opts)
    disableFont = QtGui.QFont()
    disableFont.setStrikeOut(True)
    self.enabledFontMap = {True: None, False: disableFont}
    self.item = None

  def makeTreeItem(self, depth):
    item = super().makeTreeItem(depth)
    self.enabledFontMap[True] = QtGui.QFont(item.font(0))
    item.setFont(0, self.enabledFontMap[self.opts['enabled']])
    self.item = item
    return item

  def menuActTriggered(self, act: str):
    item = self.item
    # Toggle 'enable' on click
    disabled = self.opts['enabled']
    enabled = not disabled
    item.setFont(0, self.enabledFontMap[enabled])
    for ii in range(item.childCount()):
      item.child(ii).setDisabled(disabled)
    self.opts['enabled'] = enabled

  def setOpts(self, **opts):
    enabled = opts.get('enabled', None)
    if enabled is not None and enabled != self.opts['enabled']:
      self.menuActTriggered(_toggleName)
    super().setOpts(**opts)


class FRAtomicGroupParameter(parameterTypes.GroupParameter):
  def makeTreeItem(self, depth):
    item = super().makeTreeItem(depth)
    font = QtGui.QFont()
    font.setBold(False)
    item.setFont(0, font)
    return item


class FRNoneParameter(parameterTypes.SimpleParameter):

  def __init__(self, **opts):
    opts['type'] = 'str'
    super().__init__(**opts)
    self.setWritable(False)

parameterTypes.registerParameterType('NoneType', FRNoneParameter)
parameterTypes.registerParameterType('shortcut', FRShortcutParameter)
parameterTypes.registerParameterType('procgroup', FRProcGroupParameter)
parameterTypes.registerParameterType('atomicgroup', FRAtomicGroupParameter)
parameterTypes.registerParameterType('actionwithshortcut', FRActionWithShortcutParameter)