[![pipeline status](https://gitlab.com/ficsresearch/s3a/badges/development/pipeline.svg)](https://gitlab.com/ficsresearch/s3a/-/commits/development)
[![coverage report](https://gitlab.com/ficsresearch/s3a/badges/development/coverage.svg)](https://gitlab.com/ficsresearch/s3a/-/commits/development)

# Semi-Supervised Semantic Annotator (S3A)

A highly adaptable tool for both visualizing and generating semantic annotations for generic images.

Most software solutions for semantic (pixel-level) labeling are designed for low-resolution (<10MB) images with fewer than 10 components of interest. Violating either constraint (e.g. using a high-res image or annotating ~1000 components) incur detrimental performance impacts. S3A is designed to combat both these deficiencies. With images up to 150 MB and 2000 components, the tool remains interactive.

___

A more detailed overview can be found in the project wiki [here](https://gitlab.com/ficsresearch/s3a/-/wikis/docs/user's-guide).

___

## Installation

The easiest method for installing `s3a` is via `pip` after cloning the repository:

```bash
git clone https://gitlab.com/ficsresearch/s3a
pip install -e ./s3a
```

## Running the App
Running the app is as easy as calling `s3a` as a module:
`python -m s3a`

From here, projects can be created to host groups of related images, or images can be annotated in the default project. Both options are available through the `File` menu.

## Detailed Feature List

More information about the capabilities of this tool are outlined in the [project wiki](https://gitlab.com/ficsresearch/s3a/-/wikis/home).


## License

This tool is free for personal and commercial use (except the limits imposed by the PyQt5 library). If you publish something based on results obtained through this app, please cite the following papers:

Jessurun, N., Paradis, O., Roberts, A., & Asadizanjani, N. (2020). Component Detection and Evaluation Framework (CDEF): A Semantic Annotation Tool. Microscopy and Microanalysis, 1-5. doi:10.1017/S1431927620018243

