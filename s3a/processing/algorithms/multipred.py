import copy
import functools
import inspect
import typing as t

import cv2 as cv
import numpy as np
import pandas as pd
from qtextras import FROM_PREV_IO, OptionsDict, bindInteractorOptions as bind, fns
from scipy.ndimage import maximum_filter

from .. import PipelineFunction
from ... import generalutils as gutils
from ...compio.componentio import defaultIo
from ...constants import REQD_TBL_FIELDS as RTF
from ...structures import ComplexXYVertices, XYVertices

# __all__ defined programmatically at the end of this module


def get_component_images(image: np.ndarray, components: pd.DataFrame):
    """
    From a main image and dataframe of components, returns a result which holds the
    subregion within the image each component occupies.
    """
    imgs = [
        gutils.getCroppedImage(image, verts.stack(), 0)
        for verts in components[RTF.VERTICES]
    ]
    return dict(subimages=imgs)


class ProcessDispatcher(PipelineFunction):
    def __init__(
        self,
        function,
        resultConverter: t.Callable[[t.Union[dict, t.Any], pd.Series], dict] = None,
        **kwargs,
    ):
        self.singleRunner = function
        self.resultConverter = resultConverter
        # Ignore the "component" argument, which is added by the dispatcher
        # See note below about functools.partial. If methods are encountered,
        # wrap them once more so they can have attributes set
        if inspect.ismethod(function):
            function = functools.partial(function)
        function = bind(component=dict(ignore=True))(function)

        # Something tricky: `inspect.signature` can follow __wrapped__ to get the
        # signature of the wrapped function. However, this cannot be set on
        # instance methods (like `self.dispatcher`). So, wrap in a partial
        # which *can* have a signature set.
        wrappable = functools.partial(self.dispatcher)
        super().__init__(
            functools.update_wrapper(wrappable, function),
            **kwargs,
            components=FROM_PREV_IO,
        )
        if kwargs.get("name"):
            self.__name__ = kwargs["name"]

    def dispatcher(self, components: pd.DataFrame, **kwargs):
        compList = []
        result = dict()
        for ii, comp in components.iterrows():
            kwargs.update(component=comp)
            # TODO: Determine appropriate behavior. For now, just remember last result
            #  metadata other than components
            result = self.singleRunner(**kwargs)
            if self.resultConverter is not None:
                result = self.resultConverter(result, comp)
            compList.append(result.pop("components"))
        if compList:
            outComps = gutils.concatAllowEmpty(compList, ignore_index=True)
        else:
            # Concat fails with empty list, just make an empty dataframe
            outComps = components.iloc[0:0].copy()
        outComps.index = outComps[RTF.ID]
        out = dict(**result, components=outComps)
        return out


def points_to_components(matchPts: np.ndarray, component: pd.Series):
    numOutComps = len(matchPts)
    if numOutComps == 0:
        ret = fns.seriesAsFrame(component).copy()
        return ret.loc[[]]
    # Explicit copy otherwise all rows point to the same component
    outComps = gutils.concatAllowEmpty(
        [fns.seriesAsFrame(component)] * numOutComps, ignore_index=True
    ).copy()
    origOffset = component[RTF.VERTICES].stack().min(0)
    allNewverts = []
    for ii, pt in zip(outComps.index, matchPts):
        newVerts = [verts - origOffset + pt for verts in outComps.at[ii, RTF.VERTICES]]
        allNewverts.append(ComplexXYVertices(newVerts))
    outComps[RTF.VERTICES] = allNewverts
    outComps[RTF.ID] = RTF.ID.value
    return outComps


@bind(
    threshold=dict(limits=[0, 1], step=0.1),
    metric=dict(type="list", limits=["TM_CCOEFF_NORMED"]),
    area=dict(type="list", limits=["image", "viewbox"]),
)
def cv_template_match_single(
    component: pd.Series,
    image: np.ndarray,
    viewbox: np.ndarray,
    threshold=0.8,
    metric="TM_CCOEFF_NORMED",
    area="viewbox",
):
    """
    Performs template matching using default opencv functions

    Parameters
    ----------
    component
        Template component
    image
        Main image
    threshold
        Cutoff point to consider a matched template
    metric
        Template maching metric
    area
        Where to apply the new components
    """
    template, templateBbox = gutils.getCroppedImage(
        image, component[RTF.VERTICES].stack()
    )
    if area == "viewbox":
        image, coords = gutils.getCroppedImage(image, viewbox)
    else:
        coords = np.array([[0, 0], image.shape[:2][::-1]])
    grayImg = image if image.ndim < 3 else cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    if template.ndim > 2:
        template = cv.cvtColor(template, cv.COLOR_RGB2GRAY)
    templateShp = template.shape[:2]
    if np.any(np.array(templateShp) > np.array(grayImg.shape)):
        raise ValueError(
            "Search area cannot be smaller than template size.\n"
            f"Search area size: {grayImg.shape}, template size: {templateShp}"
        )

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
        ious.append(
            gutils.boundingBoxIou(templateBbox, np.vstack([pt, pt + templateShp[::-1]]))
        )
    return dict(
        scores=scores,
        matchImg=maxFilter,
        components=points_to_components(matchPts, component),
    )


cv_template_match = ProcessDispatcher(
    cv_template_match_single, name="Cv Template Match"
)


@bind(
    area=dict(type="list", limits=["image", "viewbox"]),
    winType=dict(type="list", limits=["Row/Col Divisions", "Raw Size"]),
)
def make_grid_components(
    image: np.ndarray,
    components: pd.DataFrame,
    viewbox: np.ndarray,
    area="viewbox",
    windowParam: int = 5,
    winType="Row/Col Divisions",
    maxNumComponents=1000,
):
    """
    Creates a grid of components based on the bounding boxes of the input components

    Parameters
    ----------
    image
        Main image
    components
        Reference components, needed to determine proper output columns
    viewbox
        zoomed-in boundingBox coordinates relative to the main image
    area
        Area to apply gridding
    windowParam
        Number used during the calculation of window size. Its meaning changes
        depending on ``winType``
    winType
        If "Row/Col Divisions", the image area is divided into ``windowParam`` rows or
        columns, selcting the number resulting in a greater number of divisions. For
        instance, if ``windowParam`` is 5 and image shape is (500, 300, 3), winSize
        will be 60x60 since min(500/5, 300/5) is 60. If "Raw Size", the window size is
        directly set to ``windowParam``.
    maxNumComponents
        To prevent instances where the window parameters create too many regions,
        the number of outputs will be clipped to ``maxNumComponents``
    """
    offset = np.array([[0, 0]])
    if area == "viewbox":
        image, coords = gutils.getCroppedImage(image, viewbox)
        offset = coords[[0]]
    imageH, imageW = image.shape[:2]
    if winType == "Row/Col Divisions":
        winSize = np.min(np.asarray(image.shape[:2]) / windowParam).astype(int)
    else:
        winSize = windowParam
    winSize = max(1, winSize)
    winH = winW = winSize
    spacing = winSize
    rrange = np.arange(0, imageH, spacing)
    crange = np.arange(0, imageW, spacing)
    # Shorten number of loops by truncatnig rrange beyond max components
    rrangeTruncate = int(np.ceil(maxNumComponents / len(crange)))
    rrange = rrange[:rrangeTruncate]
    boxes = []
    for ii in rrange:
        for jj in crange:
            verts = np.array([[winW, winH]]) * [[0, 0], [0, 1], [1, 1], [1, 0]] + [
                [jj, ii]
            ]
            verts = (
                np.clip(verts, 0, [[imageW - 1, imageH - 1]])
                .astype(int)
                .view(XYVertices)
                + offset
            )
            boxes.append(ComplexXYVertices([verts]))
    boxes = boxes[:maxNumComponents]
    # Fill in other dummy fields based on passed in component dataframe fields
    df = pd.DataFrame(
        columns=[c for c in components.columns if isinstance(c, OptionsDict)]
    )
    numOutputs = len(boxes)
    for field in components:  # type: OptionsDict
        df[field] = [copy.copy(field.value) for _ in range(numOutputs)]
    df[RTF.VERTICES] = boxes
    return dict(components=df)


def merge_overlapping_components(components: pd.DataFrame):
    """
    Creates new list of components where adjacent regions are merged and everything
    is split apart after. This means every component will have at most one region
    (i.e. components that previously contained two separate regions will turn into
    two distinct components with the same metadata)

    Parameters
    ----------
    components
        Dataframe of components to merge -> split
    """
    out = dict(components=components)
    if not len(components):
        return out
    # Guarantee each component is labeled uniquely, something that is not true when
    # components are new (i.e. many have id -1)
    components = components.copy()
    dummyLabel = OptionsDict("__lbl_merge_overlap__", 0)
    components[dummyLabel] = np.arange(len(components))
    mask, mapping = defaultIo.exportLblPng(
        components, returnLabelMap=True, labelField=dummyLabel
    )
    numLbls, labels, stats, centroids = cv.connectedComponentsWithStats(
        mask.astype("uint8", copy=False)
    )
    # Use ID from each centroid as the component whose metadata is kept
    keepLabels = []
    outVerts = []
    for lbl in range(1, numLbls):
        # Greatly speed up conversion by only checking the image region which contains
        # this component
        offset = stats[lbl, [cv.CC_STAT_TOP, cv.CC_STAT_LEFT]]
        endPixs = offset + stats[lbl, [cv.CC_STAT_HEIGHT, cv.CC_STAT_WIDTH]]
        boolMaskSubset = labels[offset[0] : endPixs[0], offset[1] : endPixs[1]] == lbl
        # Keep information from first id composing the shape
        onPixs = boolMaskSubset.nonzero()
        keepLabels.append(mask[onPixs[0][0] + offset[0], onPixs[1][0] + offset[1]])
        # Offset is in row-col, removeOffset expects x-y
        outVerts.append(
            ComplexXYVertices.fromBinaryMask(boolMaskSubset).removeOffset(-offset[::-1])
        )
    oldName = components.index.name
    components["__old_index__"] = components.index
    indexable = components.set_index(dummyLabel)
    newComps = indexable.loc[mapping[keepLabels]]

    # It is possible for one old ID to point to multiple new components. To avoid
    # undefined behavior, give a new id designation to all but the first occurence
    # in these cases
    newComps.loc[newComps.index.duplicated(keep="first"), RTF.ID] = RTF.ID.value
    newComps[RTF.VERTICES] = outVerts

    # It is also possible for two components to overlap and only one ID is recycled.
    # This will mean one original component needs to be deleted. Note that components
    # which are new in the first place (i.e. from a non-refinement operation) don't need
    # to be deleted since they haven't yet been added in the first place
    maybeDeleted = components.loc[components[RTF.ID] != RTF.ID.value, newComps.columns]
    delIdxs = np.isin(maybeDeleted[RTF.ID], newComps[RTF.ID], invert=True)

    delComponents = maybeDeleted[delIdxs].copy()
    delComponents[RTF.VERTICES] = [
        ComplexXYVertices() for _ in range(len(delComponents))
    ]
    outComps = gutils.concatAllowEmpty([newComps, delComponents]).set_index(
        "__old_index__"
    )
    outComps.index.name = oldName
    return dict(components=outComps)


@bind(epsilon=dict(limits=[-1, None]))
def simplify_component_vertices(components: pd.DataFrame, epsilon=1):
    """
    Runs ``ComplexXYVertices.simplify`` on each component vertices

    Parameters
    ----------
    components
        Dataframe of components to simplify
    epsilon
        Passed to ``ComplexXYVertices.simplify``
    """
    outComps = components.copy()
    outComps[RTF.VERTICES] = [
        v.simplify(epsilon=epsilon) for v in outComps[RTF.VERTICES]
    ]
    return dict(components=outComps)


def get_selected_components(components: pd.DataFrame, selectedIds: np.ndarray):
    return dict(components=components.loc[selectedIds])


def _components_in_bounds(components: pd.DataFrame, bounds: np.ndarray):
    keepComps = []
    for idx, comp in components.iterrows():
        stacked = comp[RTF.VERTICES].stack()
        if np.all((stacked.min(0) <= bounds[1]) & (stacked.max(0) >= bounds[0])):
            # At least partially in-bounds
            keepComps.append(comp)
    # Explicitly pass columns in the event of empty "keepComps"
    return pd.DataFrame(keepComps, columns=components.columns)


@bind(overlapThreshold=dict(limits=[0, 1], step=0.1))
def remove_overlapping_components(
    components: pd.DataFrame,
    fullComponents: pd.DataFrame,
    overlapThreshold=0.5,
    removeOverlapWithExisting=True,
    removeOverlapWithNew=True,
):
    """
    Discards overlapping components. Can either check against existing components,
    other new components, or both

    Parameters
    ----------
    fullComponents
        Complete list of original components
    components
        Working list of (newly created) components
    overlapThreshold
        Percentage overlap between any new component and existing component over which
        the new component will be discarded
    removeOverlapWithExisting
        If *True*, new components overlapping with pre-existing components will be removed
    removeOverlapWithNew
        If *True*, new components overlapping with other new components will be removed
    """
    if not len(components):
        # Nothing to do...
        return dict(components=components)
    # Save computation time by only considering full components that *could* overlap
    accessor = components[RTF.VERTICES].s3averts
    newBounds = np.row_stack([accessor.min(), accessor.max()])
    fullComponents = _components_in_bounds(fullComponents, newBounds)
    if removeOverlapWithExisting and len(fullComponents):
        outShape = np.maximum(
            newBounds[1], fullComponents[RTF.VERTICES].s3averts.max()
        )[::-1]
        referenceMask = defaultIo.exportLblPng(fullComponents, imageShape=outShape)
        referenceMask[referenceMask > 0] = 1
    else:
        outShape = newBounds[1][::-1]
        referenceMask = np.zeros(outShape, "uint8")
    keepComps = []
    for idx, comp in components.iterrows():
        # Check the indexable area with a component footprint
        verts: ComplexXYVertices = comp[RTF.VERTICES]
        stacked = verts.stack()
        if not len(stacked):
            continue
        checkArea, coords = gutils.getCroppedImage(
            referenceMask, stacked, returnSlices=True, returnBoundingBox=False
        )
        # Prediction is entirely outside the image
        if checkArea.size <= 0:
            continue
        vertsMask = verts.removeOffset().toMask(checkArea.shape) > 0
        # Don't count on pixels outside the current footprint
        if (
            np.count_nonzero(checkArea & vertsMask)
            / np.count_nonzero(checkArea | vertsMask)
            < overlapThreshold
        ):
            keepComps.append(comp)
        if removeOverlapWithNew:
            # Make sure no new checks can overlap with this component
            referenceMask[coords] |= vertsMask
    # Force columns to match in the event output dataframe is empty
    return dict(components=pd.DataFrame(keepComps, columns=components.columns))


def model_prediction_factory():
    return ProcessDispatcher(categorical_prediction)


@bind(inputShape=dict(type="str"))
def single_categorical_prediction(
    component: pd.Series, image: np.ndarray, model, inputShape=None
):
    """
    Runs a single component through a model

    Parameters
    ----------
    component
        Component on which categorical mask prediction should be run
    image
        Image data to index
    model
        Model which will run prediction. If ``expectedImageShape`` is
        not specified, ``model.input_shape[1:3]`` will be used
    inputShape
        Specifies the image shape the model requires to run a prediction
    """
    inputShape = inputShape or None
    if isinstance(inputShape, str):
        inputShape = eval(inputShape, dict(model=model))
    if inputShape is not None:
        inputShape = tuple(inputShape)[:2]
    verts = component[RTF.VERTICES].stack()
    resized_image, coords, stats = gutils.subImageFromVertices(
        image,
        verts,
        returnBoundingBox=True,
        returnStats=True,
        shape=inputShape,
        interpolation=cv.INTER_NEAREST,
    )

    resized_image = np.array([resized_image])
    # Remove batch dimension
    prediction = model.predict(resized_image)[0]
    if prediction.ndim > 2:
        # 3D tensor -> assume last dimension is probs per class
        prediction = np.argmax(prediction, axis=-1)
    if prediction.ndim != 2:
        raise ValueError(
            f"Model output must be 2D or 3D tensor, got {prediction.ndim}D tensor"
        )
    prediction[prediction > 0] = 1
    prediction = gutils.inverseSubImage(
        prediction.astype("uint8"), stats, gutils.polygonToBox(verts)
    )
    if not np.any(prediction):
        return dict(components=pd.DataFrame(columns=component.index))
    out = component.copy()
    paddingOffset = verts.min(0) - stats["subImageBox"][0]
    totalOffset = -(coords[0] + paddingOffset).astype(int)
    out[RTF.VERTICES] = ComplexXYVertices.fromBinaryMask(prediction).removeOffset(
        totalOffset
    )
    return dict(components=fns.seriesAsFrame(out))


def remove_small_components(components: pd.DataFrame, sizeThreshold=30):
    outComps = []
    for idx, comp in components.iterrows():
        # Preserve empty values since they signify deletion to an outer scope
        if (
            comp[RTF.VERTICES].isEmpty()
            or np.count_nonzero(comp[RTF.VERTICES].removeOffset().toMask())
            > sizeThreshold
        ):
            outComps.append(comp)
    return dict(components=pd.DataFrame(outComps, columns=components.columns))


categorical_prediction = ProcessDispatcher(
    single_categorical_prediction, name="Categorical Prediction"
)

_selfModule = single_categorical_prediction.__module__
__all__ = [
    "categorical_prediction",
    "cv_template_match",
] + gutils.getObjectsDefinedInSelfModule(vars(), _selfModule)
