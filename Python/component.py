from constants import ComponentTypes
import numpy as np
from typing import Dict, List, Union

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
Signal = QtCore.pyqtSignal
Slot = QtCore.pyqtSlot

from SchemeEditor import SchemeEditor
from constants import SchemeValues as SV
from ABGraphics.clickables import ClickableTextItem
from ABGraphics.regions import VertexRegion

# Ensure an application instance is running
app = pg.mkQApp()

class Component(QtCore.QObject):
  _reqdUpdates: Dict[str, list] = {}

  sigCompClicked = Signal(object)

  def __init__(self):
    super().__init__()
    self.instanceId = -1
    self.vertices = np.array([np.NaN, np.NaN])
    self.deviceType = ComponentTypes.N_A
    self.boardText = ''
    self.deviceText = ''
    self.logo = ''
    self.notes = ''
    self.validated = False

    # Shorthand for convenience
    txtSize = ComponentMgr.scheme.getCompProps(SV.idFontSize)

    # Using VertexRegion is significantly faster than PlotDataItem
    #self._boundPlt = pg.PlotDataItem([np.NaN, np.NaN], pen=pg.mkPen(color=penClr, width=penWidth))
    self._boundPlt = VertexRegion()
    self._txtPlt = ClickableTextItem('N/A')
    curFont = self._txtPlt.textItem.font()
    curFont.setPointSize(txtSize)
    self._txtPlt.setFont(curFont)
    self._txtPlt.sigClicked.connect(self._rethrowItemClicked)

    '''
    IMPORTANT!! Update this list as more properties / plots are added.
    '''
    # Handles update behavior for traits that alter plot information
    self._reqdUpdates = {
      'vertices'   : [self._updateBoundPlt, self._updateTxtPlt],
      'instanceId' : [self._updateTxtPlt],
      'validated'  : [self._updateTxtPlt]
      }

  def __setattr__(self, prop, val):
    super().__setattr__(prop, val)
    # Apply plot updates depending on which variable was changed
    pltUpdateFns = self._reqdUpdates.get(prop, [])
    for fn in pltUpdateFns:
      fn()

  @staticmethod
  def setScheme(scheme: SchemeEditor):
    '''
    Responsible for customizing display aspects of each component. Only one scheme
    should exist per application, so this is a static method
    '''
    Component.scheme = scheme

  @Slot()
  def _rethrowItemClicked(self):
    self.sigCompClicked.emit(self)

  def _updateBoundPlt(self):
    #self._boundPlt.setData(self.vertices)
    self._boundPlt.updateVertices(self.vertices.copy())

  def _updateTxtPlt(self):
    schemeClrProp = SV.nonValidIdColor
    if self.validated:
      schemeClrProp = SV.validIdColor
    txtClr = Component.scheme.getCompProps(schemeClrProp)
    self._txtPlt.setText(str(self.instanceId))
    self._txtPlt.setColor(txtClr)
    newPos = np.mean(self.vertices, axis=0)
    self._txtPlt.setPos(newPos[0], newPos[1])

class ComponentMgr(QtCore.QObject):
  sigCompClicked = Signal(object)

  _compList: List[Component] = []
  _nextCompId = 0

  _compImgView: pg.ViewBox

  scheme = SchemeEditor()

  def __init__(self, mainImgArea: pg.GraphicsWidget, mainImgItem: pg.ImageItem):
    super().__init__()
    self._mainImgArea = mainImgArea
    self._mainImgItem = mainImgItem
    self._compBoundsImgItem = VertexRegion()
    #self._compBoundsImgItem = pg.PlotDataItem(connect='finite')
    # Update comp image on main image change
    mainImgItem.sigImageChanged.connect(self._updateCompBoundsImg)
    self._mainImgArea.addItem(self._compBoundsImgItem)

  def addComps(self, comps: List[Component]):
    # On first pass, collect all encountered vertices so they can be recolored
    # appropriately
    allVerts = [np.empty((0,2), dtype=int)]
    nanSep = np.empty((1,2))
    nanSep.fill(np.nan)
    for comp in self._compList: # type: Component
      allVerts.append(comp.vertices)
      #allVerts.append(nanSep)

    for comp in comps:
      allVerts.append(comp.vertices)
      #allVerts.append(nanSep)
      comp.instanceId = self._nextCompId
      #self._mainImgArea.addItem(comp._boundPlt)
      self._mainImgArea.addItem(comp._txtPlt)

      # Listen for component signals and rethrow them
      comp.sigCompClicked.connect(self._rethrowCompClick)

      self._nextCompId += 1
    self._compList.extend(comps)
    self._compBoundsImgItem.updateVertices(allVerts[1:])
    #npData = np.vstack(allVerts)
    #self._compBoundsImgItem.setData(npData[:,0], npData[:,1])

  def rmComps(self, idsToRemove: Union[np.array, str] = 'all'):
    # Use numpy array so size is preallocated
    curComps = np.array(self._compList)
    idxIsValid = np.zeros(curComps.size, dtype=bool)
    # Generate ID list
    compListIds = [obj.instanceId for obj in self._compList]
    if idsToRemove == 'all':
      idsToRemove = compListIds
    idsToRemove = np.array(idsToRemove)
    compListIds = np.array(compListIds)

    remainingVerts = []
    nanSep = np.empty((1,2))
    nanSep.fill(np.nan)
    for ii, comp in enumerate(self._compList):
      if np.any(comp.instanceId == idsToRemove):
        self._mainImgArea.removeItem(comp._txtPlt)
      else:
        remainingVerts.append(comp.vertices)
        #remainingVerts.append(nanSep)
        idxIsValid[ii] = True

    # Update bounds image
    self._compBoundsImgItem.updateVertices(remainingVerts)
    #if len(remainingVerts) > 0:
      #npData = np.vstack(remainingVerts)
    #else:
      #npData = np.empty((0,2))
    #self._compBoundsImgItem.setData(npData[:,0], npData[:,1])

    # Reset manager's component list
    self._compList = list(curComps[idxIsValid])

    # Determine next ID for new components
    newIdList = compListIds[idxIsValid]
    if newIdList.size > 0:
      self._nextCompId = np.max(newIdList) + 1
    else:
      self._nextCompId = 0

  @Slot(object)
  def _rethrowCompClick(self, comp:Component):
    self.sigCompClicked.emit(comp)

  @Slot()
  def _updateCompBoundsImg(self):
    self._compBoundsImgItem.updateVertices([])
    #self._compBoundsImgItem.setData([], [])

if __name__== '__main__':
  from PIL import Image
  mw = pg.PlotWindow()
  item = pg.ImageItem(np.array(Image.open('../fast.tif')))
  mw.addItem(item)
  mw.setAspectLocked(True)
  mw.show()

  c = Component()
  c.vertices = np.random.randint(0,100,size=(10,2))
  c.instanceId = 5
  c.boardText = 'test'
  mgr = ComponentMgr(mw, item)
  mgr.addComps([c])

  app.exec()