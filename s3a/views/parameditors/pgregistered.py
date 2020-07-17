from functools import partial
from typing import Optional, List

from pyqtgraph import TreeWidgetItem
from pyqtgraph.Qt import QtGui, QtWidgets, QtCore
from pyqtgraph.parametertree import parameterTypes, Parameter, ParameterTree
from pyqtgraph.parametertree.Parameter import PARAM_TYPES
from pyqtgraph.parametertree.parameterTypes import ActionParameterItem, ActionParameter, \
  TextParameterItem, TextParameter, WidgetParameterItem

from s3a.graphicsutils import findMainWin
from s3a.structures import FRS3AException

class FRMonkeyPatchedTextParameterItem(TextParameterItem):
  def makeWidget(self):
    textBox: QtWidgets.QTextEdit = super().makeWidget()
    textBox.setTabChangesFocus(True)
    return textBox

# Monkey patch pyqtgraph text box to allow tab changing focus
TextParameter.itemClass = FRMonkeyPatchedTextParameterItem

class FRPgParamDelegate(QtWidgets.QStyledItemDelegate):
  def __init__(self, paramDict: dict, parent=None):
    super().__init__(parent)
    errMsg = f'{self.__class__} can only create parameter editors from'
    ' registered pg widgets that implement makeWidget()'

    if paramDict['type'] not in PARAM_TYPES:
      raise FRS3AException(errMsg)
    paramDict.update(name='dummy')
    param = Parameter.create(**paramDict)
    if hasattr(param.itemClass, 'makeWidget'):
      self.item = param.itemClass(param, 0)
    else:
      raise FRS3AException(errMsg)

  def createEditor(self, parent, option, index: QtCore.QModelIndex):
    editor = self.item.makeWidget()
    editor.setParent(parent)
    editor.setMaximumSize(option.rect.width(), option.rect.height())
    return editor

  def setModelData(self, editor: QtWidgets.QWidget,
                   model: QtCore.QAbstractTableModel,
                   index: QtCore.QModelIndex):
    model.setData(index, editor.value())

  def setEditorData(self, editor: QtWidgets.QWidget, index):
    value = index.data(QtCore.Qt.EditRole)
    editor.setValue(value)

  def updateEditorGeometry(self, editor: QtWidgets.QWidget,
                           option: QtWidgets.QStyleOptionViewItem,
                           index: QtCore.QModelIndex):
    editor.setGeometry(option.rect)

class FRShortcutParameterItem(parameterTypes.WidgetParameterItem):
  """
  Class for creating custom shortcuts. Must be made here since pyqtgraph doesn't
  provide an implementation.
  """

  def makeWidget(self):
    item = QtWidgets.QKeySequenceEdit()

    item.sigChanged = item.editingFinished
    item.value = lambda: item.keySequence().toString()
    def setter(val: QtGui.QKeySequence):
      if val is None or len(val) == 0:
        item.clear()
      else:
        item.setKeySequence(val)
    item.setValue = setter
    self.param.seqEdit = item

    return item

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
    if param.opts['value'] is None:
      param.opts['value'] = ''
    shortcutSeq = param.opts['value']

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
      param.opts['value'] = newSeq.toString()
    self.keySeqEdit.keySequenceChanged.connect(updateShortcut)

    self.param.sigValueChanged.connect(lambda _, value: self.keySeqEdit.setKeySequence(value))

    # Make sure that when a parameter is removed, the shortcut is also deactivated
    param.sigRemoved.connect(lambda: self.shortcut.setParent(None))

    self.layout.addWidget(self.keySeqEdit)

class FRActionWithShortcutParameter(ActionParameter):
  itemClass = FRActionWithShortcutParameterItem

  def __init__(self, **opts):
    super().__init__(**opts)
    self.isActivateConnected = False


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