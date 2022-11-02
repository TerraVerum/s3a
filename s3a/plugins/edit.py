from __future__ import annotations

from ..constants import PRJ_CONSTS as CNST
from ..logger import getAppLogger
from ..models.s3abase import S3ABase
from .base import ParameterEditorPlugin

class EditPlugin(ParameterEditorPlugin):

    name = "Edit"

    def attachToWindow(self, window: S3ABase):
        super().attachToWindow(window)
        stack = window.actionStack

        self.registerFunction(stack.undo, name="Undo", runActionTemplate=CNST.TOOL_UNDO)
        self.registerFunction(stack.redo, name="Redo", runActionTemplate=CNST.TOOL_REDO)

        def updateUndoRedoTxts(_action=None):
            self.undoAction.setText(f"Undo: {stack.undoDescr}")
            self.redoAction.setText(f"Redo: {stack.redoDescr}")

        stack.stackChangedCallbacks.append(updateUndoRedoTxts)

        def showStatus(action):
            # Since this was the *already performed* action, what it reports is the
            # opposite of what happens
            if action is None:
                return
            if action.treatAsUndo:
                msg = f"{stack.undoDescr}"
            else:
                msg = f"Undid {stack.redoDescr}"
            getAppLogger(__name__).info(msg)

        stack.stackChangedCallbacks.append(showStatus)

        updateUndoRedoTxts()

    @property
    def undoAction(self):
        return [a for a in self.menu.actions() if a.text().startswith("Undo")][0]

    @property
    def redoAction(self):
        return [a for a in self.menu.actions() if a.text().startswith("Redo")][0]
