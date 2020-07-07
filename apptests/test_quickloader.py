from conftest import app
from s3a import FR_SINGLETON

ql = FR_SINGLETON.quickLoader

def test_normal_use():
  for editor in FR_SINGLETON.registerableEditors:
    ql.addActForEditor(editor, 'Default')
  ql.applyBtnClicked()
  # No errors should occur

def test_add_mult_times():
  ql.params.clearChildren()
  editor = FR_SINGLETON.registerableEditors[0]
  ql.addActForEditor(editor, 'Default')
  ql.addActForEditor(editor, 'Default')
  assert len(ql.params.child(editor.name).childs) == 1