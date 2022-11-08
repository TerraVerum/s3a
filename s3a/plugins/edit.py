from __future__ import annotations

from qtextras import RunOptions

from .base import ParameterEditorPlugin, ParameterEditor
from ..constants import PRJ_CONSTS as CNST
from ..logger import getAppLogger
from ..models.s3abase import S3ABase
from ..constants import PRJ_CONSTS


class EditPlugin(ParameterEditorPlugin):
    name = "Edit"
    createDock = True
    createProcessMenu = True

    def __initSharedSettings__(self, shared=None, **kwargs):
        prop = PRJ_CONSTS.PROP_UNDO_BUF_SZ
        shared.generalProperties.registerFunction(
            self.window.actionStack.resizeStack,
            parent=self.window.name,
            runOptions=RunOptions.ON_CHANGED,
            maxLength={**prop, "title": prop.name},
            nest=False,
            container=self.window.props,
        )
        shared.generalProperties.registerParameterList(
            [PRJ_CONSTS.PROP_EXP_ONLY_VISIBLE, PRJ_CONSTS.PROP_INCLUDE_FNAME_PATH],
            container=self.window.props,
            parent="Import/Export",
        )
        shared.colorScheme.registerFunction(
            self.window.updateTheme,
            runOptions=RunOptions.ON_CHANGED,
            nest=False,
            parent=self.window.name,
        )

    def attachToWindow(self, window: S3ABase):
        super().attachToWindow(window)
        # Remove "show dock" option
        self.menu.removeAction(self.menu.actions()[0])
        stack = window.actionStack

        self.registerFunction(stack.undo, name="Undo", runActionTemplate=CNST.TOOL_UNDO)
        self.registerFunction(stack.redo, name="Redo", runActionTemplate=CNST.TOOL_REDO)

        for editor in (
            window.sharedSettings.generalProperties,
            window.sharedSettings.colorScheme,
        ):
            dock, _ = editor.createWindowDock(
                window, createProcessMenu=False, addShowAction=False
            )
            showAction = self.dockRaiseAction(dock)
            showAction.setParent(self.menu)
            self.menu.addAction(showAction)

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
