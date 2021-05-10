import typing as t

import cv2 as cv
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter

from s3a.constants import PRJ_ENUMS, REQD_TBL_FIELDS as RTF
from s3a.generalutils import getCroppedImg, bboxIou
from s3a.structures import ComplexXYVertices
from utilitys import ProcessIO, fns, AtomicProcess


def get_component_images(image: np.ndarray, components: pd.DataFrame):
  """
  From a main image and dataframe of components, adds an 'img' column to `components` which holds the
  subregion within the image each component occupies.
  """
  imgs = [getCroppedImg(image, verts.stack(), 0) for verts in components[RTF.VERTICES]]
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
      newVerts = ComplexXYVertices.fromBwMask(mask)
      for v in newVerts: v += offset
      newComp[RTF.VERTICES] = newVerts
    out['components'] = fns.serAsFrame(newComp)
  out['addType'] = PRJ_ENUMS.COMP_ADD_AS_MERGE
  return out

def _dispatchFactory(func, resultConverter: t.Callable[[t.Union[dict, t.Any], pd.Series], ProcessIO]=None):
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

  proc = AtomicProcess(dispatcher, docFunc=func, ignoreKeys=['component'])
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
def _cv_template_match(component: pd.Series, image: np.ndarray, viewbox: np.ndarray,
                       threshold=0.8, metric='TM_CCOEFF_NORMED', area='viewbox',
                       refOverlapThresh=0.5):
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
    limits:
      - image
      - viewbox
  :param refOverlapThresh:
    helpText: "How much overlap can exist between the found bounding box and reference box. This helps
    prevent detecting the original as a new component"
    limits: [0,1]
    step: 0.1
  """
  template, templateBbox = getCroppedImg(image, component[RTF.VERTICES].stack())
  if area == 'viewbox':
    image, coords = getCroppedImg(image, viewbox)
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
    ious.append(bboxIou(templateBbox, np.vstack([pt, pt + templateShp[::-1]])))
  keep = np.array(ious) < refOverlapThresh
  scores = scores[keep]
  matchPts = matchPts[keep]
  return ProcessIO(scores=scores, matchImg=maxFilter, components=pts_to_components(matchPts, component))

def cv_template_match_factory():
  return _dispatchFactory(_cv_template_match)