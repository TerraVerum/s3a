from pyqtgraph.parametertree import Parameter, ParameterTree
from qtextras import ParameterEditor, fns


class MetaTreeParameterEditor(ParameterEditor):
    _metaTree: ParameterTree
    """
    Tree for controlling meta-parameters, useful for configuring the parameter 
    editor itself (i.e. loading state, etc.)
    """
    _metaParametersGroup: Parameter
    """
    Group responsible for storing the parameters that influence ``rootParameter``
    states.
    """

    def _guiChildren(self) -> list:
        self._makeMetaTree()
        outList = super()._guiChildren()
        outList.insert(outList.index(self.treeButtonsWidget) + 1, self._metaTree)
        return outList

    def _makeMetaTree(self):
        self._metaParametersGroup = Parameter.create(
            name="Meta Parameters", type="group"
        )
        self._metaTree = fns.flexibleParameterTree(
            self._metaParametersGroup, showTop=False
        )
        # self._metaParametersGroup.sigChildAdded.connect(
        #     lambda: mt.setMinimumHeight(int(mt.sizeHint().height() * 1.1))
        # )
        return self._metaTree, self._metaParametersGroup
