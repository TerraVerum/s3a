from pyqtgraph.parametertree import Parameter, ParameterTree
from qtextras import ParameterEditor, fns


class MetaTreeParameterEditor(ParameterEditor):
    _metaTree: ParameterTree
    """
    Tree for controlling meta-parameters, useful for configuring the parameter 
    editor itself (i.e. loading state, etc.)
    """
    _metaParameter: Parameter
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
        self._metaParameter = Parameter.create(name="Meta Parameters", type="group")
        self._metaTree = fns.flexibleParameterTree(self._metaParameter, showTop=False)
        # self._metaParameter.sigChildAdded.connect(
        #     lambda: mt.setMinimumHeight(int(mt.sizeHint().height() * 1.1))
        # )
        return self._metaTree, self._metaParameter
