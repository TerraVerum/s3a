[![pipeline status](https://gitlab.com/ficsresearch/s3a/badges/development/pipeline.svg)](https://gitlab.com/ficsresearch/s3a/-/commits/development)
[![coverage report](https://gitlab.com/ficsresearch/s3a/badges/development/coverage.svg)](https://gitlab.com/ficsresearch/s3a/-/commits/development)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=ficsresearch_s3a&metric=sqale_rating)](https://sonarcloud.io/dashboard?id=ficsresearch_s3a)

# Semi-Supervised Semantic Annotator (S3A)

A highly adaptable tool for both visualizing and generating semantic annotations for generic images.

Most software solutions for semantic (pixel-level) labeling are designed for low-resolution (<10MB) images with fewer than 10 components of interest. Violating either constraint (e.g. using a high-res image or annotating ~1000 components) incur detrimental performance impacts. S3A is designed to combat both these deficiencies. With images up to 150 MB and 2000 components, the tool remains interactive.

___

A more detailed overview can be found in the project wiki [here](https://gitlab.com/ficsresearch/s3a/-/wikis/docs/user's-guide).

___

## Installation

The easiest method for installing `s3a` is via `pip` after cloning the repository, or directly from pypi:

```bash
git clone https://gitlab.com/ficsresearch/s3a
pip install -e ./s3a

# Or from pypi using "pip install s3a"
```

Note that a version of OpenCV and Qt binding are required for S3A to work. These can be installed for you with the "full" option:
```bash
pip install -e ./s3a[full]
# Or "pip install s3a[full]"
```

## Running the App
Running the app is as easy as calling `s3a` as a module or using a provided entry point:
```bash
python -m s3a
```
Or, equivalently:
```bash
s3a-gui
```


From here, projects can be created to host groups of related images, or images can be annotated in the default project. Both options are available through the `File` menu.

## Detailed Feature List

More information about the capabilities of this tool are outlined in the [project wiki](https://gitlab.com/ficsresearch/s3a/-/wikis/home).

## <span style="color:red">Please Note</span>
S3A's programmatic API is still largely under development. It still needs refinement to allow for consistent naming schemes, removing vestigial elements, confirming private vs. public-facing elements, and a few other line items. However, the graphical interface should be minimally affected by these alterations.

Thus, while the GUI entry point should be consistently useful, be aware of these developments when using the scripting portion of S3A in great detail.

## License

This tool is free for personal and commercial use (except the limits imposed by the PyQt5 library). If you publish something based on results obtained through this app, please cite the following papers:

Jessurun, N., Paradis, O., Roberts, A., & Asadizanjani, N. (2020). Component Detection and Evaluation Framework (CDEF): A Semantic Annotation Tool. Microscopy and Microanalysis, 1-5. doi:10.1017/S1431927620018243

