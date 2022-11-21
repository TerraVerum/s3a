import copy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from qtextras import OptionsDict, fns
from qtextras.typeoverloads import FilePath

from .templatemgr import IOTemplateManager
from .yamlparser import YamlParser
from ..constants import REQD_TBL_FIELDS as RTF
from ..generalutils import getMaybeReplaceKey
from ..parameditors.tablefilter import TableFilterEditor
from ..structures import OptionsDictGroup


def getFieldAliases(field: OptionsDict):
    """
    Returns the set of all potential aliases to a given field
    """
    return set([field.name] + field.opts.get("aliases", []))


def aliasesToRequired(field: OptionsDict):
    """
    Returns true or false depending on whether this field shares aliases with required
    fields. This is useful when an alternative (incoming) representation of e.g. Vertices
    must be suppressed on import, but still used on export
    """
    requiredAliases = set()
    for reqdField in RTF:
        requiredAliases.update(getFieldAliases(reqdField))
    srcAliases = getFieldAliases(field)
    return srcAliases & requiredAliases


class TableData(QtCore.QObject):
    sigConfigUpdated = QtCore.Signal(object)
    """dict (self.config) during update"""
    filter: Optional[TableFilterEditor]

    ioTemplate = "s3a"

    def __init__(
        self,
        configPath: FilePath = None,
        configDict: dict = None,
        template: Union[FilePath, dict] = None,
        makeFilter=False,
    ):
        super().__init__()
        if template is None:
            template = self.ioTemplate
        if isinstance(template, str):
            template = IOTemplateManager.getTableConfig(template)
        self.template = template

        self.factories: Dict[OptionsDict, Callable[[], Any]] = {}

        if makeFilter:
            self.filter = TableFilterEditor()
        else:
            self.filter = None
        self.parameterParser: Optional[YamlParser] = None

        self.configPath: Optional[Path] = None
        self.config: Optional[dict] = {}

        self.allFields: List[OptionsDict] = []
        self.resetLists()

        configPath = configPath or None
        self.loadConfig(configPath, configDict, force=True)

    def makeComponentDf(self, rows=1, sequentialIds=False) -> pd.DataFrame:
        """
        Creates a dataframe for the requested number of components. This is the
        recommended method for component instantiation prior to table insertion.
        """
        df_list = []
        dropRow = False
        if rows <= 0:
            # Create one row and drop it, which ensures data types are correct in the
            # empty dataframe
            rows = 1
            dropRow = True
        populators = []
        for f in self.allFields:
            if f in self.factories:
                val = self.factories[f]()
            else:
                val = f.value
            populators.append(val)

        for _ in range(rows):
            # Make sure to construct a separate component instance for
            # each row no objects have the same reference
            df_list.append(copy.copy(populators))
        outDf = pd.DataFrame(df_list, columns=self.allFields)
        if RTF.ID in self.allFields:
            if sequentialIds:
                outDf[RTF.ID] = np.arange(len(outDf), dtype=int)
            outDf = outDf.set_index(RTF.ID, drop=False)
        if dropRow:
            outDf = outDf.iloc[0:0]
        return outDf

    def addFieldFactory(self, fieldLabel: OptionsDict, factory: Callable[[], Any]):
        """
        For fields that are simple functions (i.e. don't require input from the user),
        a factory can be used to create default values when instantiating new table rows.

        Parameters
        ----------
        fieldLabel
            WHich field this factory is used for instead of just the default value
        factory
            Callable to use instead of field value. This is called with no parameters.
        """
        self.factories[fieldLabel] = factory

    def addField(self, field: OptionsDict):
        """
        Adds a new field to the table. If the field already exists in the current
        table, no action is performed. Returns *True* if a field really was added,
        *False* if this field is already in the table list or aliases to an existing
        field
        """

        # Problems occur when fields alias to already existing ones. When this is the
        # case, ignore the extra fields. Not only does this solve the many-to-one alias
        # issue, but also allows table datas with different required fields to
        # seamlessly share and swap fields with eachother while avoiding vestigial
        # table columns
        if field in self.allFields or self._findMatchingField(field) is not field:
            return False
        field.group = self.allFields
        self.allFields.append(field)
        if field.name not in self.config["fields"]:
            # Added programmatically outside config, ensure file representation is not
            # lost
            self.config["fields"][field.name] = newFieldCfg = dict(field)
            # Remove redundant `name` field
            newFieldCfg.pop("name")
        return True

    def makeComponentSeries(self):
        return self.makeComponentDf().squeeze()

    def loadConfig(
        self, configPath: FilePath = None, configDict: dict = None, force=False
    ):
        """
        Lodas the specified table configuration file for S3A. Alternatively, a name
        and dict pair can be supplied instead.

        Parameters
        ----------
        configPath
            If *configDict* is *None*, this is treated as the file containaing a
            YAML-compatible table configuration dictionary. Otherwise, this is the
            configuration name assiciated with the given dictionary.
        configDict
            If not *None*, this is the config data used instad of reading ``configFile``
            as a file.
        force
            If *True*, the new config will be loaded even if it is the same name as the
            current config
        """
        baseConfigDict = copy.deepcopy(self.template)
        if configPath is not None:
            configPath, configDict = fns.resolveYamlDict(configPath, configDict)
            configPath = configPath.resolve()
        # Often, a table config can be wrapped in a project config; look for this case
        # first
        if configDict is not None and (
            "table-config" in configDict or "table-cfg" in configDict
        ):
            configDict = getMaybeReplaceKey(
                configDict, oldKey="table-cfg", newKey="table-config"
            )

        fns.hierarchicalUpdate(baseConfigDict, configDict, uniqueListElements=True)
        cfg = baseConfigDict
        if not force and self.configPath == configPath and pg.eq(cfg, self.config):
            return None

        self.configPath = configPath or self.configPath
        self.config = cfg
        self.parameterParser = YamlParser(cfg)
        self.resetLists()
        for field in cfg.get("fields", {}):
            param = self.parameterParser["fields", field]
            self.addField(param)

        if self.filter:
            self.filter.updateParameterList(self.allFields)
        self.sigConfigUpdated.emit(self.config)

    def clear(self):
        self.loadConfig(configDict={})

    def resetLists(self):
        self.allFields.clear()

    def fieldFromName(self, name: Union[str, OptionsDict], default=None):
        """
        Helper function to retrieve the OptionsDict corresponding to the field with this
        name
        """
        return OptionsDictGroup.fieldFromParameter(self.allFields, name, default)

    def resolveFieldAliases(self, fields: Sequence[OptionsDict], mapping: dict = None):
        """
        Several forms of imports / exports handle data that may not be compatible with
        the current table data. In these cases, it is beneficial to determine a mapping
        between names to allow greater compatibility between I/O formats. Mapping is
        also extended in both directions by parameter name aliases (parameter.opts[
        'aliases']), which are a list of strings of common mappings for that parameter
        (e.g. [Class, Label] are often used interchangeably)

        Parameters
        ----------
        fields
            Dataframe with maybe foreign fields
        mapping
            Foreign to local field name mapping
        """

        outFields = []
        for srcField in fields:
            outFields.append(self._findMatchingField(srcField, mapping))
        return outFields

    def _findMatchingField(self, srcField, mapping: dict = None):
        # Mapping takes priority, if it exists
        if mapping is None:
            mapping = {}
        potentialSrcNames = getFieldAliases(srcField)
        for key in srcField, srcField.name:
            # Mapping can either be by string or OptionsDict, so account for either case
            outCol = mapping.get(key)
            if outCol:
                break

        if outCol is not None:
            # A mapping was explicitly provided for this field, use that
            return self.fieldFromName(outCol)
        elif srcField in self.allFields:
            return srcField
        else:
            # Not in mapping, no exact match.
            # TODO: what if multiple dest cols have a matching alias?
            #   Otherwise, a 'break' can be added
            curOutName = srcField
            for destField in self.allFields:
                if potentialSrcNames & getFieldAliases(destField):
                    # Match between source field's aliases and dest field aliases Make
                    # sure it didn't match multiple names that weren't itself with the
                    # assert statement In other words, if multiple dest fields have the
                    # same alias, this assertion will fail
                    assert curOutName == srcField
                    curOutName = destField
        return curOutName

    def __reduce__(self):
        return TableData, (
            self.configPath,
            self.config,
            self.template,
            self.filter is not None,
        )
