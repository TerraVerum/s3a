# --------------
# Errors used within the application
# --------------
class CsvIOError(Exception): pass
class InvalidDrawModeError(Exception): pass
class IllRegisteredPropError(Exception): pass
class ParamParseError(Exception): pass

from pyqtgraph.Qt import QtWidgets
class _MainWin(QtWidgets.QMainWindow): pass