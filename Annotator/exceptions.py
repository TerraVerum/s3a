# --------------
# Errors used within the application
# --------------
class FRCsvIOError(Exception): pass
class FRInvalidDrawModeError(Exception): pass
class FRIllRegisteredPropError(Exception): pass
class FRParamParseError(Exception): pass
class FRInvalidROIEvType(Exception): pass
class FRIllFormedVertices(Exception): pass

from pyqtgraph.Qt import QtWidgets
class _MainWin(QtWidgets.QMainWindow): pass