from argparse import Action

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from qtextras import fns

# from . import __version__, mkQApp
from . import mkQApp
from .constants import PRJ_ENUMS
from .views.s3agui import S3A


def main(loadLastState=True, **load):
    """
    Calling code for the S3A application.

    Parameters
    ----------
    loadLastState
        When the app is closed, all settings are saved. If this is *True*,
        these settings are restored on startup. If *False*, they aren't.
    load
        States to load, see the help output for possible values
    """
    mkQApp()
    win = S3A(log=PRJ_ENUMS.LOG_GUI, loadLastState=loadLastState, **load)
    QtCore.QTimer.singleShot(0, win.showMaximized)
    pg.exec()
    return win


def mainCli():
    parser = fns.makeCli(main, parserKwargs=dict(prog="S3A", add_help=False))
    parser.register("action", "help", S3AHelp)
    parser.add_argument("--version", action="version", version="TODO")
    parser.add_argument("--help", action="help")
    args = parser.parse_args()
    main(**vars(args))


class S3AHelp(Action):
    def __init__(self, **kwargs):
        kwargs.update(nargs=0)
        super().__init__(**kwargs)

    def __call__(self, parser, *args, **kwargs) -> None:
        mkQApp()
        win = S3A(loadLastState=False, log=PRJ_ENUMS.LOG_NONE)
        win.updateCliOptions(parser)
        # Prevent settings overwrite
        win.appStateEditor.saveParameterValues = lambda *args, **kwargs: None
        win.forceClose()
        parser.print_help()
        parser.exit()


if __name__ == "__main__":
    mainCli()
