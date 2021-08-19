import copy
import typing as t

from utilitys import fns
from utilitys.typeoverloads import FilePath
from ...constants import IO_TEMPLATES_DIR
from ...generalutils import DirectoryDict


class IOTemplateManager:
  """
  Handles required table data for a given I/O type outside S3A.
  For instance, VGG Image Annotator requires `region_shape_attributes` instead of `Vertices`, and is parsed
   much differently. This is the entry point for fetching such required information
  that will exist regardless of the metadata associated with a specific VGG project. If no defaults are required
  or this type hasn't been registered, an `None` is returned.
  """
  templates = DirectoryDict(IO_TEMPLATES_DIR, fns.attemptFileLoad, allowAbsolute=True)

  @classmethod
  def getTableCfg(cls, ioType: str):
    if ioType is None:
      return None
    cfgFname = ioType.lower() + '.tblcfg'
    cfg = cls.templates.get(cfgFname)
    if cfg is None:
      return None
    return copy.deepcopy(cfg)

  @classmethod
  def registerTableCfg(cls, ioType: str, cfg: t.Union[FilePath, dict], force=False):
    """
    Associates the given field information as containing required fields for `ioType`. Either an absolute filepath to a
    configuration or a dictionary of `fields: {}` data can be passed. These are fields that must exist in the io format
    """
    key = ioType.lower() + '.tblcfg'
    # DirectoryDict doesn't yet implement 'contains', so just try to get the key first
    try:
      ret = cls.templates[key]
    except KeyError:
      # Key doesn't exist no need to do anything
      pass
    else:
      if not force:
        raise KeyError(f'I/O type "{ioType}" already has associated data and `force` is false:\n{ret}')

    if isinstance(cfg, FilePath.__args__):
      cfg = fns.attemptFileLoad(cfg)
    cls.templates[key] = cfg
