import pytest

from s3a.structures import S3AWarning, S3AException
from s3a.parameditors import *

ql = FR_SINGLETON.quickLoader
editor = FR_SINGLETON.colorScheme
def test_normal_add():
  ql.addActForEditor(editor, 'Default')
  ql.applyChanges()
  assert editor.name in ql.params.names
  ql.params.clearChildren()

def test_double_add():
  ql.addActForEditor(editor, 'Default')
  ql.addActForEditor(editor, 'Default')
  assert editor.name in ql.params.names
  param = ql.params.child(editor.name)
  assert len(param.children()) == 1
  ql.params.clearChildren()

@pytest.mark.qt_no_exception_capture
def test_invalid_load(qtbot):
  # Pytest isn't catching this error correctly for some reason, try wrapping in caller
  # function within qt event loop
  def invalidLoadCaller():
    ql.addActForEditor(editor, 'SaveOptionThatDoesntExist')
    ql.applyChanges()
  with qtbot.capture_exceptions() as exceptions:
    invalidLoadCaller()
    foundException = False
    for ex in exceptions:
      if issubclass(ex[0], S3AException):
        foundException = True
        break
    assert foundException
  assert len(ql.params.child(editor.name).children()) == 0
  ql.params.clearChildren()

def test_from_line_edit():
  ql.addNewParamState.setText(ql.listModel.displayFormat.format(stateName='Default',
                                                                editor=editor))
  ql.addFromLineEdit()
  assert len(ql.params.child(editor.name).children()) == 1
  ql.params.clearChildren()

def test_invalid_line_edit_add():
  ql.addNewParamState.setText('Doesnt Exist')
  ql.addFromLineEdit()
  assert len(ql.params.children()) == 0

def test_bad_user_profile():
  invalidFileDict = {'colorscheme': 'doesnt exist'}
  with pytest.warns(S3AWarning):
    ql.buildFromUserProfile(invalidFileDict)

def test_bad_load_state(qtbot):
  badLoad = dict(name='Non-existent editor', type='group',
                 children=[dict(name='bad action', type='actionwithshortcut', value='Test')])
  pstate = dict(name='test', type='group', children=[badLoad])
  with pytest.warns(S3AWarning):
    ql.loadParamState('bad state', pstate)
