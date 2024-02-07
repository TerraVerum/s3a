from __future__ import annotations

from abc import ABC
from typing import List, Optional, Type

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from qtextras import OptionsDict, bindInteractorOptions as bind

from .regions import MultiRegionPlot
from ..constants import REQD_TBL_FIELDS as RTF
from ..generalutils import getTopLeftCoordinate, symbolFromVertices, concatAllowEmpty
from ..structures import OptionsDictGroup


class FieldDisplayDelegate(ABC):
    def addToDisplay(self, plotItem: pg.PlotItem):
        """
        Responsible for adding its own displayable information to the plotItem
        """
        raise NotImplementedError

    def removeFromDisplay(self, plotItem: pg.PlotItem):
        """
        Responsible for removing its own displayable information from the plotItem
        """
        raise NotImplementedError

    def clear(self):
        """Clear all displayed data"""
        raise NotImplementedError

    def show(self):
        """Show existing data"""
        raise NotImplementedError

    def hide(self):
        """Hide data without clearing it"""
        raise NotImplementedError

    def setData(self, components: pd.DataFrame, field: OptionsDict, **kwargs):
        """
        Called with all data that matches this delegate.

        Parameters
        ----------
        components
            DataFrame of all components that match this delegate
        field
            Field to which this delegate belongs. In some cases, data *outside* this
            field is required for proper display (i.e. text labels need Vertices for
            their position). So, while the whole dataframe is passed, only the data
            that is relevant to this delegate should be consumed. No matter what,
            `components` should *not* be modified.
        **kwargs
            ways to customise the display. `setData` will be registered as a function in
          a ParamEditor, so these will be user-customizable.
        """
        raise NotImplementedError


class SceneItemContainer(FieldDisplayDelegate):
    """
    This base class handles all cases where all visual representations are scene items,
    which already support show, hide, removeItem, addItem. So, classes which inherit
    this must indicate a list of `item`s on which these operations should be called
    """

    items: List[pg.GraphicsObject] = None
    plotItem: Optional[pg.PlotItem] = None

    def show(self):
        for item in self.items:
            item.show()

    def hide(self):
        for item in self.items:
            item.hide()

    def removeFromDisplay(self, plotItem: pg.PlotItem):
        for item in self.items:
            plotItem.removeItem(item)
        self.plotItem = None

    def addToDisplay(self, plotItem: pg.PlotItem):
        for item in self.items:
            plotItem.addItem(item)
        self.plotItem = plotItem

    # Many items support 'clear', so provide a default implementation when this is the
    # case
    def clear(self):
        for item in self.items:
            item.clear()


class TextFieldDelegate(SceneItemContainer):
    def __init__(self):
        super().__init__()
        self.scatter = pg.ScatterPlotItem()
        self.bgScatter = pg.ScatterPlotItem()
        self.scatter.setZValue(100)
        self.bgScatter.setZValue(99)
        self.items = [self.scatter, self.bgScatter]

    @bind(textColor=dict(type="color"))
    def setData(
        self,
        components: pd.DataFrame,
        field: OptionsDict,
        fontSize=10,
        textColor="w",
        **kwargs,
    ):
        positions = np.row_stack(
            components[RTF.VERTICES].apply(lambda el: getTopLeftCoordinate(el.stack()))
        )
        symbols_scales = np.row_stack(
            components[field].apply(self.makeTextSymbol, returnScale=True)
        )
        keepLocs = [s is not None for s in symbols_scales[:, 0]]
        positions = positions[keepLocs]
        symbols_scales = symbols_scales[keepLocs]
        bgSymbols = [
            self.makeBgSymbol(symb) for symb in symbols_scales[:, 0] if symb is not None
        ]
        multiplier = fontSize / 12
        sizes = multiplier / symbols_scales[:, 1]

        self.scatter.setData(
            *positions.T,
            pen=textColor,
            brush=textColor,
            symbol=symbols_scales[:, 0],
            size=sizes,
        )
        self.bgScatter.setData(
            *positions.T, symbol=bgSymbols, size=sizes * 1.1, pen="w", brush="#000a"
        )
        self.scatter.invalidate()
        self.bgScatter.invalidate()

    @staticmethod
    def makeBgSymbol(textSymbol):
        path = QtGui.QPainterPath()
        rect = textSymbol.boundingRect()
        path.addRect(rect)
        return path

    @staticmethod
    def makeTextSymbol(text: str, fontSize=12, returnScale=False):
        if not text:
            # No way to draw symbol
            if returnScale:
                return None, 1
            else:
                return None
        outSymbol = QtGui.QPainterPath()
        textLabel = QtGui.QFont("Sans Serif", fontSize)
        # textLabel.setStyleStrategy(QtGui.QFont.StyleStrategy.PreferBitmap)
        lines = text.split("\n")
        outSymbol.addText(0, 0, textLabel, lines[0])
        height = outSymbol.boundingRect().height()
        heightWithMargin = height * 1.2
        for ii, line in enumerate(lines[1:]):
            outSymbol.addText(0, heightWithMargin * (ii + 1), textLabel, line)
        br = outSymbol.boundingRect()
        scale = 1.0 / max(br.width(), br.height())
        tr = QtGui.QTransform()
        # Somehow the transform flips on the x/y axis, so counteract with negative scale
        tr.scale(scale, scale)
        tr.translate(-br.x() - br.width() / 2.0, -br.y() - br.height() / 2.0)
        outSymbol = tr.map(outSymbol)
        if returnScale:
            return outSymbol, scale
        return outSymbol


class XYVerticesDelegate(SceneItemContainer):
    def __init__(self):
        self.polyScatter = pg.ScatterPlotItem()
        self.polyScatter.setPxMode(False)
        self.polyScatter.setSize(1)

        self.pointScatter = pg.ScatterPlotItem()
        self.pointScatter.setPxMode(True)

        self.calloutCurve = pg.PlotCurveItem()

        self.items = [self.polyScatter, self.pointScatter, self.calloutCurve]
        for it in self.items:
            it.setZValue(100)

    @bind(spotColor=dict(type="color"))
    def setData(
        self,
        components: pd.DataFrame,
        field: OptionsDict,
        spotSize=15,
        spotColor="y",
        calloutWidth=1,
        **kwargs,
    ):
        # Single coordinates and disconnected vertices are points, while connected
        # vertices are shaded polygons.
        polyVerts, pointVerts, callouts = [], [], []
        for idx, verts in components[field].items():
            # Ignore empty vertices
            if not len(verts):
                continue
            # Hard to see a polygon with only 2 points
            if verts.connected and len(verts) > 2:
                # Close the shape
                verts = np.row_stack([verts, verts[[0]]])
                polyVerts.append(verts)
            else:
                pointVerts.append(verts)
            point = verts[[0]]
            callouts.append(
                self._calloutLine(components.at[idx, RTF.VERTICES].stack(), point)
            )

        # Checking length prevents "need at least one array to concatenate" error
        if len(polyVerts):
            polySymbols, polyPositions = zip(
                *[symbolFromVertices(_verts) for _verts in polyVerts]
            )
            polyPositions = np.row_stack(polyPositions)
        else:
            polySymbols = "o"  # Doesn't matter
            polyPositions = np.ones((0, 2))
        self.polyScatter.setData(
            *polyPositions.T, pen=spotColor, brush=None, symbol=polySymbols
        )

        if len(pointVerts):
            pointPositions = np.row_stack(pointVerts)
        else:
            pointPositions = np.ones((0, 2))
        self.pointScatter.setData(
            *pointPositions.T, pen=spotColor, brush=spotColor, size=spotSize
        )

        if calloutWidth <= 0 or not len(callouts):
            # Draw a line from each spot / polygon to the nearest component vertex
            self.calloutCurve.clear()
        else:
            calloutPositions = np.row_stack(callouts)
            pen = pg.mkPen(color=spotColor, width=calloutWidth)
            self.calloutCurve.setData(*calloutPositions.T, pen=pen, connect="pairs")

    @staticmethod
    def _calloutLine(vertices, point):
        """
        Creates a callout line from the point to the comp vertex closest to the origin
        """
        return np.row_stack([getTopLeftCoordinate(vertices), point])


class ComplexXYVerticesDelegate(SceneItemContainer):
    LBL_PARAM = OptionsDict("complex delegate label")

    def __init__(self):
        self.region = MultiRegionPlot(disableMouseClick=True)
        self.items = [self.region]

    def setData(self, components: pd.DataFrame, field: OptionsDict, **kwargs):
        setComps = pd.DataFrame()
        setComps[RTF.VERTICES] = components[field]
        # TODO: Expose label parameter for coloring, etc.
        self.region.resetRegionList(setComps)


class FieldDisplay:
    """
    Handles the display of component field data.
    """

    DEFAULT_FIELD = OptionsDict(
        "stringified display data", "", type="uniqueval_default"
    )

    def __init__(self, plotItem: pg.PlotItem):
        self.availableDelegates = {}
        self.plotItem = plotItem
        self.inUseDelegates = []
        self.defaultDelegate = TextFieldDelegate
        self.registerDelegate(
            self.DEFAULT_FIELD.type, self.defaultDelegate, override=True
        )
        self.registerDelegate("xyvertices", XYVerticesDelegate, override=True)
        self.registerDelegate(
            "complexxyvertices", ComplexXYVerticesDelegate, override=True
        )
        # Vertices are already displayed, source image is the same for all components
        self.ignoreColumns = [RTF.VERTICES, RTF.IMAGE_FILE]

    def registerDelegate(
        self, fieldType, delegate: Type[FieldDisplayDelegate], override=False
    ):
        """
        Assigns a delegate for displaying a type of data. I.e. a MultiRegionPlot is the
        delegate for vertices-like data. `fieldType` corresponds to the same field as
        OptionsDict.fieldType, which is a registered form of field data. It may be a single
        string or tuple of strings, each denoting a type (i.e. 'int', 'list', etc.).
        """
        if not isinstance(fieldType, tuple):
            fieldType = (fieldType,)
        for typ in fieldType:
            if typ in self.availableDelegates and not override:
                raise ValueError(f"Delegate for {typ} already registered")
            self.availableDelegates[typ] = delegate

    def showFieldData(self, components: pd.DataFrame, fields=None, **kwargs):
        self.callDelegateFunction("clear")
        if not len(components):
            return

        allFields = list(components)
        if fields is None:
            fields = allFields
        else:
            # user arguments could be strings
            fields = [OptionsDictGroup.fieldFromParameter(allFields, f) for f in fields]

        defaultFields = [
            f
            for f in fields
            if self.availableDelegates.get(f.type, self.defaultDelegate)
            is self.defaultDelegate
            and f not in self.ignoreColumns
        ]

        if defaultFields:
            components = self._replaceDefaultData(components, defaultFields)
            # Ensure defaults don't get thrown out below
            fields.append(self.DEFAULT_FIELD)

        for field in components:
            if field in self.ignoreColumns or field not in fields:
                continue
            delegateCls = self.availableDelegates.get(field.type, None)
            if delegateCls is None:
                continue
            delegate = delegateCls()
            delegate.addToDisplay(self.plotItem)
            delegate.setData(components, field, **kwargs)
            self.inUseDelegates.append(delegate)

    @classmethod
    def _replaceDefaultData(cls, components: pd.DataFrame, fields):
        """
        Concats all data into text for display with the default delagate, and drops the
        original fields
        """
        converter = lambda comp: "\n".join(
            f"{index}: {data}" for index, data in comp.items() if len(str(data))
        )
        text = components[fields].apply(converter, axis=1)
        text.name = cls.DEFAULT_FIELD
        keepColumns = np.setdiff1d(components.columns, fields)
        delegateComps = concatAllowEmpty([text, components[keepColumns]], axis=1)
        return delegateComps

    def callDelegateFunction(self, functionName, *args):
        for delegate in self.inUseDelegates:
            getattr(delegate, functionName)(*args)
        # Special case: clearing means letting go of refs
        if functionName == "clear":
            self.callDelegateFunction("removeFromDisplay", self.plotItem)
            self.inUseDelegates.clear()
