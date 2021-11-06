import copy
import typing as t

import cv2 as cv
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter

from ...constants import PRJ_ENUMS, REQD_TBL_FIELDS as RTF
from ... import generalutils as gutils
from ...structures import ComplexXYVertices, XYVertices
from ...compio.componentio import defaultIo
from utilitys import PrjParam
from utilitys import ProcessIO, fns, AtomicProcess


def get_component_images(image: np.ndarray, components: pd.DataFrame):
  """
  From a main image and dataframe of components, returns a result which holds the
  subregion within the image each component occupies.
  """
  imgs = [gutils.getCroppedImg(image, verts.stack(), 0) for verts in components[RTF.VERTICES]]
  return ProcessIO(subimages=imgs)

def _focusedResultConverter(result, component: pd.Series):
  out = result
  if not isinstance(result, ProcessIO):
    out = ProcessIO(image=result)
  if 'components' not in out:
    mask = out['image']
    newComp = component.copy()
    if mask is not None:
      offset = newComp[RTF.VERTICES].stack().min(0)
      newVerts = ComplexXYVertices.fromBinaryMask(mask)
      for v in newVerts: v += offset
      newComp[RTF.VERTICES] = newVerts
    out['components'] = fns.serAsFrame(newComp)
  out['addType'] = PRJ_ENUMS.COMP_ADD_AS_MERGE
  return out

def _dispatchFactory(
  func,
  resultConverter: t.Callable[[t.Union[dict, t.Any], pd.Series], ProcessIO]=None,
  name=None,
):
  def dispatcher(image: np.ndarray, components: pd.DataFrame, **kwargs):
    compList = []
    kwargs.update(image=image)
    result = ProcessIO()
    for ii, comp in components.iterrows():
      kwargs.update(component=comp)
      # TODO: Determine appropriate behavior. For now, just remember last result metadata other than comps
      result = func(**kwargs)
      if resultConverter is not None:
        result = resultConverter(result, comp)
      compList.append(result.pop('components'))
    if compList:
      # Concat fails with empty list
      outComps = pd.concat(compList, ignore_index=True)
    else:
      outComps = components.drop(components.index).copy()
    out = ProcessIO(**result, components=outComps)
    return out

  proc = AtomicProcess(dispatcher, name, docFunc=func, ignoreKeys=['component'])
  proc.input['components'] = ProcessIO.FROM_PREV_IO
  return proc

def pts_to_components(matchPts: np.ndarray, component: pd.Series):
  numOutComps = len(matchPts)
  if numOutComps == 0:
    ret = fns.serAsFrame(component).copy()
    return ret.loc[[]]
  # Explicit copy otherwise all rows point to the same component
  outComps = pd.concat([fns.serAsFrame(component)]*numOutComps, ignore_index=True).copy()
  origOffset = component[RTF.VERTICES].stack().min(0)
  allNewverts = []
  for ii, pt in zip(outComps.index, matchPts):
    newVerts = [verts-origOffset+pt for verts in outComps.at[ii, RTF.VERTICES]]
    allNewverts.append(ComplexXYVertices(newVerts))
  outComps[RTF.VERTICES] = allNewverts
  return outComps


@fns.dynamicDocstring(metricTypes=[d for d in dir(cv) if d.startswith('TM')])
def _cv_template_match(
  component: pd.Series,
  image: np.ndarray,
  viewbox: np.ndarray,
  threshold=0.8,
  metric='TM_CCOEFF_NORMED',
  area='viewbox'
):
  """
  Performs template matching using default opencv functions
  :param component: Template component
  :param image: Main image
  :param threshold:
    helpText: Cutoff point to consider a matched template
    limits: [0, 1]
    step: 0.1
  :param metric:
    helpText: Template maching metric
    pType: list
    limits: {metricTypes}
  :param area:
    helpText: Where to apply the new components
    pType: list
    limits: ['image', 'viewbox']
  """
  template, templateBbox = gutils.getCroppedImg(image, component[RTF.VERTICES].stack())
  if area == 'viewbox':
    image, coords = gutils.getCroppedImg(image, viewbox)
  else:
    coords = np.array([[0,0], image.shape[:2][::-1]])
  grayImg = image if image.ndim < 3 else cv.cvtColor(image, cv.COLOR_RGB2GRAY)
  if template.ndim > 2:
    template = cv.cvtColor(template, cv.COLOR_RGB2GRAY)
  templateShp = template.shape[:2]
  if np.any(np.array(templateShp) > np.array(grayImg.shape)):
    raise ValueError('Search area cannot be smaller than template size.\n'
                     f'Search area size: {grayImg.shape}, template size: {templateShp}')

  metric = getattr(cv, metric)
  res = cv.matchTemplate(grayImg, template, metric)
  maxFilter = maximum_filter(res, templateShp)
  # Non-max suppression to remove close-together peaks
  res[maxFilter > res] = 0
  loc = np.nonzero(res >= threshold)
  scores = res[loc]
  matchPts = np.c_[loc[::-1]] + coords[[0]]
  # Don't allow matches on top of originals
  ious = []
  for pt in matchPts:
    ious.append(gutils.bboxIou(templateBbox, np.vstack([pt, pt + templateShp[::-1]])))
  return ProcessIO(scores=scores, matchImg=maxFilter, components=pts_to_components(matchPts, component))

def cv_template_match_factory():
  return _dispatchFactory(_cv_template_match)

def make_grid_components(
  image: np.ndarray,
  components: pd.DataFrame,
  viewbox: np.ndarray,
  area='viewbox',
  windowParam: int=5,
  winType='Row/Col Divisions',
  maxNumComponents=1000
):
  """
  :param image: Main image
  :param components: Reference components, needed to determine proper output columns
  :param viewbox: zoomed-in bbox coordinates relative to the main image
  :param area: Area to apply gridding
  type: list
  limits: ['image', 'viewbox']
  :param windowParam: Number used during the calculation of window size. Its meaning changes depending on ``winType``
  :param winType: If "Row/Col Divisions", the image area is divided into ``windowParam`` rows or columns,
    selcting the number resulting in a greater number of divisions. For instance, if ``windowParam`` is 5 and
    image shape is (500, 300, 3), winSize will be 60x60 since min(500/5, 300/5) is 60. If "Raw Size",
    the window size is directly set to ``windowParam``.
    type: list
    limits: ['Row/Col Divisions', 'Raw Size']
  :param maxNumComponents: To prevent instances where the window parameters create too many regions, the number
    of outputs will be clipped to ``maxNumComponents``
  """
  offset = np.array([[0, 0]])
  if area == 'viewbox':
    image, coords = gutils.getCroppedImg(image, viewbox)
    offset = coords[[0]]
  imageH, imageW = image.shape[:2]
  if winType == 'Row/Col Divisions':
    winSize = np.min(np.asarray(image.shape[:2]) / windowParam).astype(int)
  else:
    winSize = windowParam
  winSize = max(1, winSize)
  winH = winW = winSize
  spacing = winSize
  rrange = np.arange(0, imageH, spacing)
  crange = np.arange(0, imageW, spacing)
  # Shorten number of loops by truncatnig rrange beyond max components
  rrangeTruncate = int(np.ceil(maxNumComponents/len(crange)))
  rrange = rrange[:rrangeTruncate]
  boxes = []
  for ii in rrange:
    for jj in crange:
      verts = np.array([[winW, winH]]) * [[0, 0], [0, 1], [1, 1], [1, 0]] + [[jj, ii]]
      verts = np.clip(verts, 0, [[imageW - 1, imageH - 1]]).astype(int).view(XYVertices) + offset
      boxes.append(ComplexXYVertices([verts]))
  boxes = boxes[:maxNumComponents]
  # Fill in other dummy fields based on passed in component dataframe fields. Assume input df has PrjParam headers
  # since it should've come from a prediction input
  df = pd.DataFrame(columns=components.columns)
  numOutputs = len(boxes)
  for field in components: # type: PrjParam
    df[field] = [copy.copy(field.value) for _ in range(numOutputs)]
  df[RTF.VERTICES] = boxes
  return ProcessIO(components=df)

def simplify_components(
  components: pd.DataFrame
):
  """
  Simplifies a list of components by merging adjacent/overlapping regions
  :param components: Dataframe of components to simplify
  """
  if not len(components):
    outComps = components
  else:
    merged = components[RTF.VERTICES].s3averts.merge()
    newVerts = pd.Series([merged], index=[components.index[0]]).s3averts.split()
    outComps = components.loc[newVerts.index].copy()
    outComps[RTF.VERTICES] = newVerts
  return ProcessIO(components=outComps)

def get_selected_components(components: pd.DataFrame, selectedIds: np.ndarray):
  return ProcessIO(components=components.loc[selectedIds])

def remove_overlapping_components(
  components: pd.DataFrame,
  fullComponents: pd.DataFrame,
  overlapThreshold=0.5,
  removeOverlapWithExisting=True,
  removeOverlapWithNew=True
):
  """
  Discards overlapping components. Can either check against existing components, other new components, or both
  :param fullComponents: Complete list of original components
  :param components: Working list of (newly created) components
  :param overlapThreshold: Percentage overlap between any new component and existing component over which the new
    component will be discarded
    limits: [0,1]
    step: 0.1
  :param removeOverlapWithExisting: If *True*, new components overlapping with pre-existing components will be removed
  :param removeOverlapWithNew: If *True*, new components overlapping with other new components will be removed
  """
  if not len(components):
    # Nothing to do...
    return ProcessIO(components=components)
  if removeOverlapWithExisting:
    outShape = pd.concat([fullComponents, components], ignore_index=True)[RTF.VERTICES].s3averts.max()[::-1]
    referenceMask = defaultIo.exportLblPng(fullComponents, imageShape=outShape)
    referenceMask[referenceMask > 0] = 1
  else:
    outShape = components[RTF.VERTICES].s3averts.max()[::-1]
    referenceMask = np.zeros(outShape, 'uint8')
  keepComps = []
  for idx, comp in components.iterrows():
    # Check the indexable area with a component footprint
    verts: ComplexXYVertices = comp[RTF.VERTICES]
    checkArea, coords = gutils.getCroppedImg(referenceMask, verts.stack(), coordsAsSlices=True)
    vertsMask = verts.removeOffset().toMask(checkArea.shape)
    # Don't count on pixels outside the current footprint
    if np.count_nonzero(checkArea & vertsMask)/checkArea.size < overlapThreshold:
      keepComps.append(comp)
    if removeOverlapWithNew:
      # Make sure no new checks can overlap with this component
      referenceMask[coords] |= vertsMask
  # Force columns to match in the event output dataframe is empty
  return ProcessIO(components=pd.DataFrame(keepComps, columns=components.columns))

def model_prediction_factory():
  return _dispatchFactory(categorical_prediction)

def single_categorical_prediction(
  component: pd.Series,
  image: np.ndarray,
  model,
  expectedImageShape=None
):
  """
  :param component: Component on which categorical mask prediction should be run
  :param image: Image data to index
  :param model: Model which will run prediction. If ``expectedImageShape`` is not specified, ``model.input_shape[1:3]``
    will be used
  :param expectedImageShape: Specifies the image shape the model requires to run a prediction
    ignore: True
  """
  if expectedImageShape is None:
    expectedImageShape = model.input_shape[1:3]
  verts = component[RTF.VERTICES].stack()
  resized_image, coords, stats = gutils.subImageFromVerts(
    image, verts,
    returnCoords=True,
    returnStats=True,
    shape=expectedImageShape[:2],
    interpolation=cv.INTER_NEAREST
  )

  resized_image = np.array([resized_image])
  prediction = model.predict(resized_image)
  prediction = np.argmax(prediction[0], axis = -1)
  prediction[prediction > 0] = 1
  prediction = gutils.inverseSubImage(prediction.astype('uint8'), stats, gutils.coordsToBbox(verts))
  out = component.copy()
  paddingOffset = verts.min(0) - stats['subImageBbox'][0]
  totalOffset = -(coords[0] + paddingOffset).astype(int)
  out[RTF.VERTICES] = ComplexXYVertices.fromBinaryMask(prediction).removeOffset(totalOffset)
  #return np.random.random((image.shape[:2])) > 0
  return ProcessIO(components=fns.serAsFrame(out), addType=PRJ_ENUMS.COMP_ADD_AS_MERGE)

categorical_prediction = _dispatchFactory(
  single_categorical_prediction,
  name='Categorical Prediction'
)
