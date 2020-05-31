# --------------
# Errors used within the application
# --------------
class FRCdefException(Exception): pass
class FRAppIOError(IOError, FRCdefException): pass
class FRInvalidDrawModeError(FRCdefException): pass
class FRIllRegisteredPropError(FRCdefException): pass
class FRParamParseError(FRCdefException): pass
class FRInvalidROIEvType(FRCdefException): pass
class FRIllFormedVerticesError(FRCdefException): pass
class FRAlgProcessorError(FRCdefException): pass
class FRUndoStackError(FRCdefException): pass