from argparse import Action

from pyqtgraph.Qt import QtCore

from s3a.constants import PRJ_ENUMS
from utilitys import fns
from . import __version__
from . import appInst
from .views.s3agui import S3A


def main(loadLastState=True, **load):
  """
  Calling code for the S3A application.

  :param loadLastState: When the app is closed, all settings are saved. If this is *True*,
    these settings are restored on startup. If *False*, they aren't.
  :param load: States to load, see the help output for possible values
  """
  # Handle here for faster bootup
  win = S3A(log=PRJ_ENUMS.LOG_GUI, loadLastState=loadLastState, **load)
  QtCore.QTimer.singleShot(0, win.showMaximized)
  try:
    appInst.exec_()
  except AttributeError:
    appInst.exec()
  return win

def main_cli():
  parser = fns.makeCli(main, parserKwargs=dict(prog='S3A', add_help=False))
  parser.register('action', 'help', S3AHelp)
  parser.add_argument('--version', action='version', version=__version__)
  parser.add_argument('--help', action='help')
  args = parser.parse_args()
  main(**vars(args))

class S3AHelp(Action):
  def __init__(self, **kwargs):
    kwargs.update(nargs=0)
    super().__init__(**kwargs)

  def __call__(self, parser, *args, **kwargs) -> None:
    win = S3A(loadLastState=False, log=PRJ_ENUMS.LOG_NONE)
    win.makeHelpOpts(parser)
    # Prevent settings overwrite
    win.appStateEditor.saveParamValues = lambda *args, **kwargs: None
    win.forceClose()
    parser.print_help()
    parser.exit()


if __name__ == '__main__':
  main_cli()