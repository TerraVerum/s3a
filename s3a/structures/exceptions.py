# --------------
# Errors used within the application
# --------------
class S3AException(Exception): pass
class S3AWarning(Warning): pass

class S3AIOError(IOError, S3AException): pass
class InvalidDrawModeError(S3AException): pass
class ParamEditorError(S3AException): pass
class InvalidROIEvType(S3AException): pass
class IllFormedVerticesError(S3AException): pass
class AlgProcessorError(S3AException): pass
class ActionStackError(S3AException): pass