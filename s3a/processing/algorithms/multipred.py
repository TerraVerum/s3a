import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter

from s3a.constants import PRJ_ENUMS, REQD_TBL_FIELDS as RTF
from s3a.generalutils import getCroppedImg
from s3a.structures import ComplexXYVertices
from s3a.processing import GlobalPredictionProcess
from utilitys import ProcessIO, fns, AtomicProcess

import cv2 as cv

def get_component_images(image: np.ndarray, components: pd.DataFrame):
  """
  From a main image and dataframe of components, adds an 'img' column to `components` which holds the
  subregion within the image each component occupies.
  """
  imgs = [getCroppedImg(image, verts.stack(), 0) for verts in components[RTF.VERTICES]]
  return ProcessIO(subimages=imgs)


def _dispatchedTemplateMatcher(func):
  def dispatcher(image: np.ndarray, components: pd.DataFrame, **kwargs):
    out = ProcessIO()
    allComps = []
    for ii, comp in components.iterrows():
      verts = comp[RTF.VERTICES].stack()
      template = getCroppedImg(image, verts, 0, returnSlices=False)
      result = func(image=image, template=template, **kwargs)
      if isinstance(result, ProcessIO):
        pts = result['matchPts']
        out.update(**result)
        out.pop('components', None)
      else:
        pts = result
      allComps.append(pts_to_components(pts, comp))
    outComps = pd.concat(allComps, ignore_index=True)
    out['components'] = outComps
    out['deleteOrig'] = True
    return out

  proc = AtomicProcess(dispatcher, docFunc=func, ignoreKeys=['template'])
  proc.input['components'] = ProcessIO.FROM_PREV_IO
  return proc


def _dispatchedFocusedProcessor(func):
  def dispatcher(image: np.ndarray, components: pd.DataFrame, **kwargs):
    out = ProcessIO()
    allComps = []
    for ii, comp in components.iterrows():
      verts = comp[RTF.VERTICES].stack()
      focusedImage = getCroppedImg(image, verts, 0, returnSlices=False)
      result = func(image=focusedImage, **kwargs)
      if isinstance(result, ProcessIO):
        mask = result['image']
        out.update(**result)
        out.pop('components', None)
      else:
        mask = result
      newComp = comp.copy()
      if mask is not None:
        newComp[RTF.VERTICES] = ComplexXYVertices.fromBwMask(mask)
      allComps.append(fns.serAsFrame(newComp))
    outComps = pd.concat(allComps)
    out['components'] = outComps
    out['addType'] = PRJ_ENUMS.COMP_ADD_AS_MERGE
    return out
  proc = GlobalPredictionProcess()
  proc.addFunction(dispatcher, docFunc=func)
  proc.stages[0].input['components'] = ProcessIO.FROM_PREV_IO
  return proc


def pts_to_components(matchPts: np.ndarray, component: pd.Series):
  numOutComps = len(matchPts)
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
def _cv_template_match(template: np.ndarray, image: np.ndarray, viewbox: np.ndarray,
                      threshold=0.8, metric='TM_CCOEFF_NORMED', area='viewbox'):
  """
  Performs template matching using default opencv functions
  :param template: Template image
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
    limits:
      - image
      - viewbox
  """
  if area == 'viewbox':
    image, coords = getCroppedImg(image, viewbox, 0)
  else:
    coords = np.array([[0,0]])
  grayImg = image if image.ndim < 3 else cv.cvtColor(image, cv.COLOR_RGB2GRAY)
  if template.ndim > 2:
    template = cv.cvtColor(template, cv.COLOR_RGB2GRAY)

  metric = getattr(cv, metric)
  res = cv.matchTemplate(grayImg, template, metric)
  maxFilter = maximum_filter(res, template.shape[:2])
  # Non-max suppression to remove close-together peaks
  res[maxFilter > res] = 0
  loc = np.nonzero(res >= threshold)
  scores = res[loc]
  matchPts = np.c_[loc[::-1]] + coords[[0]]
  return ProcessIO(matchPts=matchPts, scores=scores, matchImg=maxFilter)

def cv_template_match_factory():
  return _dispatchedTemplateMatcher(_cv_template_match)