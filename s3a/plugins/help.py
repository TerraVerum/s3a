from __future__ import annotations

from pyqtgraph import QtWidgets, QtGui, QtCore
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
        htmlStr = """
    <div>Icons made by <a href="https://www.freepik.com" title="Freepik">Freepik</a> from <a href="https://www.flaticon.com/" title="Flaticon">www.flaticon.com</a></div>

    <div>Icons made by <a href="https://www.flaticon.com/authors/those-icons" title="Those Icons">Those Icons</a> from <a href="https://www.flaticon.com/" title="Flaticon">www.flaticon.com</a></div>

    <div>Icons made by <a href="https://www.flaticon.com/authors/pixel-perfect" title="Pixel perfect">Pixel perfect</a> from <a href="https://www.flaticon.com/" title="Flaticon">www.flaticon.com</a></div>
    
    <div>Icons made by <a href="https://www.flaticon.com/authors/google" title="Google">Google</a> from <a href="https://www.flaticon.com/" title="Flaticon">www.flaticon.com</a></div>
    """

        QtWidgets.QMessageBox.information(self.win, "Icon Attributions", htmlStr)
