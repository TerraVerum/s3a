import logging

import pytest


@pytest.fixture()
def ql(app):
    return app.sharedSettings.quickLoader


@pytest.fixture()
def editor(app):
    return app.sharedSettings.generalProperties


def test_normal_add(ql, editor):
    ql.addActionForEditor(editor, "Default")
    assert editor.name in ql.rootParameter.names
    ql.rootParameter.clearChildren()


def test_double_add(ql, editor):
    ql.addActionForEditor(editor, "Default")
    ql.addActionForEditor(editor, "Default")
    assert editor.name in ql.rootParameter.names
    param = ql.rootParameter.child(editor.name)
    assert len(param.children()) == 1
    ql.rootParameter.clearChildren()


@pytest.mark.qt_no_exception_capture
def test_invalid_load(caplog, ql, editor):
    # Pytest isn't catching this error correctly for some reason, try wrapping in caller
    # function within qt event loop
    def invalidLoadCaller():
        ql.addActionForEditor(editor, "SaveOptionThatDoesntExist")
        ql.loadFirstStateFromEachEditor()

    invalidLoadCaller()
    crits = [r for r in caplog.records if r.levelno == logging.CRITICAL]
    assert crits
    assert len(ql.rootParameter.child(editor.name).children()) == 0
    ql.rootParameter.clearChildren()
    caplog.clear()


def test_from_line_edit(ql, editor):
    ql.addNewEditorState.setText(
        ql.listModel.displayFormat.format(stateName="Default", editor=editor)
    )
    ql.addFromLineEdit()
    assert len(ql.rootParameter.child(editor.name).children()) == 1
    ql.rootParameter.clearChildren()


def test_invalid_line_edit_add(ql):
    ql.addNewEditorState.setText("Doesnt Exist")
    ql.addFromLineEdit()
    assert len(ql.rootParameter.children()) == 0


def test_bad_user_profile(ql):
    invalidFileDict = {"colorscheme": "doesnt exist"}
    with pytest.warns(UserWarning):
        ql.buildFromStartupParameters(invalidFileDict)


def test_load_state(ql):
    state = {
        "Color Scheme": {"Default": None},
        "Vertices Processor": {"Default": None},
        "App Settings": {"Default": None},
    }
    ql.loadParameterValues("test", state)
    assert len(ql.rootParameter.children()) == 3
    for ch in ql.rootParameter:
        assert "Default" in ch.names


#
def test_bad_load_state(ql):
    state = {
        "Nonsense": {"Default": None},
    }
    with pytest.raises(ValueError):
        ql.loadParameterValues("test", state)
