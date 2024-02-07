from __future__ import annotations

import inspect
import typing as t
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from qtextras import FROM_PREV_IO, OptionsDict, ParameterEditor, fns
from qtextras.typeoverloads import FilePath

from .helpers import checkVerticesBounds, deserialize
from ..constants import REQD_TBL_FIELDS as RTF
from ..generalutils import toDictGen, concatAllowEmpty
from ..structures import AnnInstanceError, AnnParseError
from ..tabledata import TableData


class _GenericExportProtocol(t.Protocol):
    def __call__(
        self, componentDf: pd.DataFrame, exportObject, **kwargs
    ) -> (t.Any, pd.DataFrame):
        return exportObject, NO_ERRORS.copy()


class _updateExportObjectProtocol(t.Protocol):
    def __call__(self, inst: dict, exportObject, **kwargs) -> t.Any:
        return exportObject


# Alias just for better readability
NO_ERRORS = pd.DataFrame()

_exportCallable = t.Callable[[pd.DataFrame, t.Any], t.Tuple[t.Any, pd.DataFrame]]


class AnnotationIOBase:
    __name__: t.Optional[str] = None
    """
    Determines which config template's required fields are necessary for this input. 
    That way, required fields don't have to be explicitly enumerated in a project's 
    table configuration
    """

    def __init__(self, options=None):
        """
        Provides access to a modularized version of the common import structure:
          * read a file
          * parse bulk columns, where applicable (one to one column mapping)
          * parse individual instances, where applicable (one to many or many to
            one column mapping)
          * apply formatting

        This is all viewable under the ``__call__`` function.

        Parameters
        ----------
        options
            Dict-like metadata for this importer/exporter. If *None*, defaults to empty
            option set. This will be updated with kwargs from being called.
        """
        # Compatibility with function analysis done in ComponentIO
        clsName = type(self).__name__
        prefix = "import" if "Importer" in clsName else "export"
        fmtName = type(self).__name__.replace("Importer", "").replace("Exporter", "")
        self.__name__ = self.__name__ or f"{prefix}{fmtName}"

        if options is None:
            options = {}
        self.options = options

    def populateMetadata(self, **kwargs):
        return self._forwardMetadata(**kwargs)

    @classmethod
    def optionsMetadata(cls):
        """
        Get all metadata descriptions from self and any base class ``populateMetadata``.
        """
        metadata = {}
        classes = [
            curcls
            for curcls in inspect.getmro(cls)
            if issubclass(curcls, AnnotationIOBase)
        ]
        # Reverse so most current class is last to override options
        for subcls in reversed(classes):
            parsed = ParameterEditor.defaultInteractor.functionToParameterDict(
                subcls.populateMetadata, title=fns.nameFormatter
            )
            curMeta = {
                ch["name"]: ch
                for ch in parsed["children"]
                if not ch.get("ignore", False)
                and ch.get("value") is not FROM_PREV_IO
                and not ch["name"].startswith("_")
            }
            metadata.update(curMeta)
        return metadata

    def _forwardMetadata(self, locals_=None, **kwargs):
        """
        Convenience function to update __call__ kwargs from some locals and extra
        keywords, since this is a common paradigm in `populateMetadata`
        """
        if locals_ is None:
            locals_ = {}
        keySource = {**locals_, **kwargs}

        useKeys = set(kwargs).union(self.optionsMetadata())
        # Can only populate requested keys if they exist in the keysource
        return {kk: keySource[kk] for kk in useKeys.intersection(keySource)}


class AnnotationExporter(AnnotationIOBase):
    exportObject: t.Any
    componentDf: t.Optional[pd.DataFrame] = None

    bulkExport: _GenericExportProtocol | None = None
    """
    Can be defined if bulk-exporting (whole dataframe at once) is possible. Must
    accept inputs (component dataframe, export object, **kwargs) and output
    tuple[export object, error dataframe]. If no errors, error dataframe should
    be present but empty.
    """

    updateExportObject: _updateExportObjectProtocol | None = None
    """
    Can be defined if individual importing (row-by-row) is possible. This is fed
    the current dataframe row as a dict of cell values and is expected to output the 
    updated export object (which will be passed to writeFile). Must accept inputs
    (instance dict, export object, **kwargs) and output the export object.
    """

    class ERROR_COL:
        pass

    """Sentinel class to add errors to an explanatory message during export"""

    def writeFile(self, file: FilePath, exportObject, **kwargs):
        raise NotImplementedError

    def createExportObject(self, **kwargs):
        raise NotImplementedError

    def individualExport(self, componentDf: pd.DataFrame, exportObject, **kwargs):
        """
        Returns an export object + dataframe of row + errors, if any occurred for some
        rows
        """
        if self.updateExportObject is None:
            # Can't do anything, don't modify the object and save time not iterating
            # over rows
            return exportObject, NO_ERRORS.copy()
        errs = []
        for row in toDictGen(componentDf):
            try:
                exportObject = self.updateExportObject(row, exportObject, **kwargs)
            except Exception as err:
                row[self.ERROR_COL] = err
                errs.append(row)
        return exportObject, pd.DataFrame(errs)

    def formatReturnObject(self, exportObject, **kwargs):
        # If metadata options change return behavior, that can be resolved here.
        return exportObject

    def __call__(
        self,
        componentDf: pd.DataFrame,
        file: FilePath = None,
        errorOk=False,
        **kwargs,
    ):
        file = Path(file) if isinstance(file, FilePath.__args__) else None
        self.componentDf = componentDf

        kwargs.update(file=file)
        activeOpts = {**self.options, **kwargs}
        meta = self.populateMetadata(**activeOpts)
        kwargs.update(**meta)

        exportObject = self.createExportObject(**kwargs)
        for func in (
            self.bulkExport,
            self.individualExport,
        ):  # type: _GenericExportProtocol
            if func is None:
                continue
            exportObject, errs = func(componentDf, exportObject, **kwargs)
            if len(errs) and not errorOk:
                raise ValueError(
                    "Encountered problems exporting the following annotations:\n"
                    + errs.to_string()
                )
        self.exportObject = exportObject
        if file is not None:
            self.writeFile(kwargs.pop("file"), exportObject, **kwargs)
        toReturn = self.formatReturnObject(exportObject, **kwargs)
        self.componentDf = None
        return toReturn


class AnnotationImporter(AnnotationIOBase):
    ioTemplate = "s3a"

    importObject: t.Any

    formatSingleInstance = None
    """
    Can be defined to cause row-by-row instance parsing. If defined, must accept
    inputs (instance dict, **kwargs) and output a dict of instance values.
    """

    bulkImport = None
    """
    Can be defined to parse multiple traits from the imported object into a component 
    dataframe all at once. Must accept inputs (import object, **kwargs) and output a
    dataframe of instance values. Note that in some cases, a direct conversion of 
    instances to a dataframe is convenient, so ``defaultBulkImport`` is provided for 
    these cases. Simply set bulkImport = ``AnnotationImporter.defaultBulkImport`` if 
    you wish.
    """

    def __init__(self, tableData: TableData = None):
        """
        Provides access to a modularized version of the common import structure:

          * read a file
          * parse bulk columns, where applicable (one to one column mapping)
          * parse individual instances, where applicable (one to many or many to one
            column mapping)
          * apply formatting

        This is all viewable under the `__call__` function.

        Parameters
        ----------
        tableData
            Table configuration dictating how each metadata field is parsed, i.e.
            converting "True"/"False" to booleans, etc. If not provided, all fields
            will be parsed as strings.
        """

        super().__init__()
        self.tableData = tableData

    def readFile(self, file: FilePath, **kwargs):
        raise NotImplementedError

    def getInstances(self, importObject, **kwargs):
        raise NotImplementedError

    def finalizeImport(self, componentDf, **kwargs):
        """Deserializes any columns that are still strings"""
        if not len(componentDf):
            return componentDf.copy()
        fields = self.tableData.allFields if self.tableData else []

        outDf = pd.DataFrame()
        # If tableData or column spec are provided, attempt to serialize as needed.
        # Otherwise, assume all fields are strings and leave them as is.
        for col in componentDf.columns:
            dfVals = componentDf[col]
            if col in fields:
                # get OptionsDict version that knows how to deserialize
                col = fields[fields.index(col)]
            if isinstance(col, OptionsDict):
                # Serialize with native option from column
                dfVals, parsedErrs = deserialize(col, dfVals)
                parsedErrs = parsedErrs.apply(AnnInstanceError)
                dfVals = concatAllowEmpty([dfVals, parsedErrs])
            # Else, assume field should stay as-is
            outDf[col] = dfVals
        # All recognized output fields should now be deserialied
        return outDf

    def __call__(
        self,
        inputFileOrObject: t.Union[FilePath, t.Any],
        *,
        parseErrorOk=False,
        reindex=False,
        keepExtraFields=None,
        addMissingFields=False,
        **kwargs,
    ):
        """
        Imports a file or dataframe object by converting string data representations
        into S3A-compatible types (like :class:`ComplexXYVertices`, etc.).

        Parameters
        ----------
        inputFileOrObject
            File path or object to import
        parseErrorOk
            If True, rows in the table with parsing errors will be silently removed.
            Otherwise, an error will be raised.
        reindex
            If True, the index of the returned dataframe will be reset to range(n) where
            n is the number of rows. Otherwise, the index will be preserved. Note that
            ``RTF.ID`` is used as the index, so if this column is not present, the index
            will be reset regardless of this parameter.
        keepExtraFields
            If True, any fields in the input file that are not recognized by the
            :class:`TableData` will be kept in the output dataframe. If False, they will
            be removed. If None, will be ``True`` if ``self.tableData` is populated
            else ``False``.
        addMissingFields
            If True, any fields in the :class:`TableData` that are not present in the
            input file will be added to the output dataframe with their default values.
            Otherwise, no additional action is performed.
        **kwargs
            Additional keyword arguments to pass to the import function.
        """
        if keepExtraFields is None:
            keepExtraFields = self.tableData is not None
        if self.tableData is None and (keepExtraFields or addMissingFields):
            warnings.warn(
                "Specifying `keepExtraFields` or `addMissingFields` while "
                "`self.tableData` is None will have no effect.",
                RuntimeWarning,
                stacklevel=2,
            )
        file = (
            Path(inputFileOrObject)
            if isinstance(inputFileOrObject, FilePath.__args__)
            else None
        )
        if file is not None:
            inputFileOrObject = self.readFile(inputFileOrObject, **kwargs)
        self.importObject = inputFileOrObject

        kwargs.update(file=file, reindex=reindex)
        activeOpts = {**self.options, **kwargs}
        meta = self.populateMetadata(**activeOpts)
        kwargs.update(meta)

        parsedDfs = []
        for func in (
            self.individualImport,
            self.bulkImport,
        ):  # type: t.Callable[[t.Any, ...], pd.DataFrame]
            # Default to empty dataframes for unspecified importers
            if func is None:
                func = lambda *_args, **_kw: pd.DataFrame()
            parsedDfs.append(func(inputFileOrObject, **kwargs))

        indivParsedDf, bulkParsedDf = parsedDfs
        # Overwrite bulk-parsed information with individual if needed, or add to it
        bulkParsedDf[indivParsedDf.columns] = indivParsedDf
        # Some cols could be deserialized, others could be serialized still. Handle the
        # still serialized cases
        parsedDf = self.finalizeImport(bulkParsedDf, **kwargs)

        if not keepExtraFields and self.tableData:
            keepCols = [col for col in parsedDf if col in self.tableData.allFields]
            parsedDf = parsedDf[keepCols]

        if addMissingFields and self.tableData:
            # False positive SettingWithCopyWarning occurs if missing fields were added
            # and the df was reordered, but copy() is not a performance bottleneck
            # and at least grants the new `parsedDf` explicit ownership of its data
            parsedDf = parsedDf.copy()

            # Destination fields that never showed up should be appended
            for field in self.tableData.allFields:
                # Special case: instance id is handled below
                if field not in parsedDf and field != RTF.ID:
                    parsedDf[field] = field.value

        # Make sure IDs are present
        parsedDf = self._ensureIdsAsIndex(parsedDf, reindex=reindex)

        # Now that all column names and settings are resolve, handle any bad imports
        validDf = self.validInstances(parsedDf, parseErrorOk)
        # Ensure reindexing still takes place if requested, some instances were invalid,
        # and invalid instances were allowed to be dropped (`parseErrorOk`)
        if reindex and len(validDf) != len(parsedDf):
            validDf[RTF.ID] = validDf.index = np.arange(len(validDf))

        # Ensure vertices present, optionally check against known image shape
        if "imageShape" in kwargs and RTF.VERTICES in validDf:
            checkVerticesBounds(validDf[RTF.VERTICES], kwargs.get("imageShape"))
        return validDf

    @staticmethod
    def _ensureIdsAsIndex(df, reindex=None):
        alreadyExists = RTF.ID in df
        inserIndex = 0 if alreadyExists else len(df.columns)
        if reindex or not alreadyExists:
            sequentialIds = np.arange(len(df), dtype=int)
            if alreadyExists:  # Just reindexing
                df[RTF.ID] = sequentialIds
            # Ensure instance ID is the first column if new
            else:
                df.insert(0, RTF.ID, sequentialIds)
        elif not pd.api.types.is_integer_dtype(df[RTF.ID]):
            # pandas 1.4 introduced FutureWarnings for object-dtype assignments so ensure
            # Instance ID is integer type
            df[RTF.ID] = df[RTF.ID].astype(int)
        return df.set_index(RTF.ID, drop=False)

    @classmethod
    def validInstances(cls, parsedDf: pd.DataFrame, parseErrorOk=False):
        errIdxs = parsedDf.apply(
            lambda row: any(isinstance(vv, AnnInstanceError) for vv in row), axis=1
        ).to_numpy(bool)
        if not np.any(errIdxs):
            return parsedDf
        if not parseErrorOk:
            raise AnnParseError(instances=parsedDf, invalidIndexes=errIdxs)
        # If only a subset is kept, `copy` is necessary to avoid SettingsWithCopyWarning
        # when this df is modified elsewhere
        return parsedDf[~errIdxs].copy()

    def defaultBulkImport(self, importObject, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(self.getInstances(importObject, **kwargs))

    def individualImport(self, importObject, **kwargs):
        parsed = []
        if self.formatSingleInstance is None:
            return pd.DataFrame()
        for inst in self.getInstances(importObject, **kwargs):
            parsedInst = self.formatSingleInstance(inst, **kwargs)
            parsed.append(parsedInst)

        indivParsedDf = pd.DataFrame(parsed)
        return indivParsedDf
