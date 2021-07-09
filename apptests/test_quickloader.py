import logging

import pytest

from conftest import assertExInList

@pytest.fixture()
def ql(app):
  return app.sharedAttrs.quickLoader

@pytest.fixture()
def editor(app):
  return app.sharedAttrs.generalProps

def test_normal_add(ql, editor):
  ql.addActForEditor(editor, 'Default')
  ql.applyChanges()
  assert editor.name in ql.params.names
  ql.params.clearChildren()

def test_double_add(ql, editor):
  ql.addActForEditor(editor, 'Default')
  ql.addActForEditor(editor, 'Default')
  assert editor.name in ql.params.names
  param = ql.params.child(editor.name)
  assert len(param.children()) == 1
  ql.params.clearChildren()

@pytest.mark.qt_no_exception_capture
def test_invalid_load(caplog, ql, editor):
  # Pytest isn't catching this error correctly for some reason, try wrapping in caller
  # function within qt event loop
  def invalidLoadCaller():
    ql.addActForEditor(editor, 'SaveOptionThatDoesntExist')
    ql.applyChanges()
  invalidLoadCaller()
  crits = [r for r in caplog.records if r.levelno == logging.CRITICAL]
  assert crits
  assert len(ql.params.child(editor.name).children()) == 0
  ql.params.clearChildren()
  caplog.clear()

def test_from_line_edit(ql, editor):
  ql.addNewParamState.setText(ql.listModel.displayFormat.format(stateName='Default',
                                                                editor=editor))
  ql.addFromLineEdit()
  assert len(ql.params.child(editor.name).children()) == 1
  ql.params.clearChildren()

def test_invalid_line_edit_add(ql):
  ql.addNewParamState.setText('Doesnt Exist')
  ql.addFromLineEdit()
  assert len(ql.params.children()) == 0

def test_bad_user_profile(ql):
  invalidFileDict = {'colorscheme': 'doesnt exist'}
  with pytest.warns(UserWarning):
    ql.buildFromStartupParams(invalidFileDict)

def test_load_state(ql):
  state = {
    'Color Scheme': {'Default': None},
    'Vertices Processor': {'Default': None},
    'App Settings': {'Default': None}
  }
  ql.loadParamValues('test', state)
  assert len(ql.params.childs) == 3
  for ch in ql.params:
    assert 'Default' in ch.names
#
def test_bad_load_state(ql):
  state = {
    'Nonsense': {'Default': None},
  }
  with pytest.raises(ValueError):
    ql.loadParamValues('test', state)
