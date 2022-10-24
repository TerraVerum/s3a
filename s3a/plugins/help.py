from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from utilitys import ParamEditorPlugin


class HelpPlugin(ParamEditorPlugin):
    name = "Help"

    def attachWinRef(self, win: QtWidgets.QMainWindow):
        super().attachWinRef(win)
        self.registerFunc(
            lambda: QtGui.QDesktopServices.openUrl(
                QtCore.QUrl(
                    "https://gitlab.com/ficsresearch/s3a/-/wikis/docs/user's-guide"
                )
            ),
            name="Online User Guide",
        )
        self.registerFunc(
            lambda: QtWidgets.QMessageBox.aboutQt(win, "About Qt"), name="About Qt"
        )
        self.registerFunc(self.iconAttributionsGui, name="Icon Attributions")

    def iconAttributionsGui(self):
        flaticonUrl = "<a href='https://www.flaticon.com/'>flaticon</a>"

        htmlStr = f"""
            <div>Icons made by <a href="https://www.freepik.com">Freepik</a> 
            from {flaticonUrl}</div>
            
            <div>Icons made by <a href="https://www.flaticon.com/authors/those-icons">
            Those Icons</a> from {flaticonUrl}</div>

            <div>Icons made by <a href="https://www.flaticon.com/authors/pixel-perfect">
            Pixel perfect</a> from {flaticonUrl}</div>
    
            <div>Icons made by <a href="https://www.flaticon.com/authors/google"> 
            Google</a> from {flaticonUrl}</div> 
        """

        QtWidgets.QMessageBox.information(self.win, "Icon Attributions", htmlStr)
