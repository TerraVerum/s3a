[![pipeline status](https://gitlab.com/ficsresearch/s3a/badges/development/pipeline.svg)](https://gitlab.com/ficsresearch/s3a/-/commits/development)
[![coverage report](https://gitlab.com/ficsresearch/s3a/badges/development/coverage.svg)](https://gitlab.com/ficsresearch/s3a/-/commits/development)

# Semi-Supervised Semantic Annotator (S3A)

A highly adaptable tool for both visualizing and generating semantic annotations for generic images.

Most software solutions for semantic (pixel-level) labeling are designed for low-resolution (<10MB) images with fewer than 10 components of interest. Violating either constraint (e.g. using a high-res image or annotating ~1000 components) incur detrimental performance impacts. S3A is designed to combat both these deficiencies. With images up to 150 MB and 2000 components, the tool remains interactive. However, since the use case is tailored to multiple small regions of interest within an image, performance lags when editing individual components of larger than ~1000x1000 pixels. However, for simple bounding polygons (i.e. no running of semantic segmentation algorithms), the individual component sizes can be arbitrarily large.

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

However, if this is the first time you are starting S3A, you will run into the following error message:
```
No author name provided and no default author exists. Exiting.
To start without error, provide an author name explicitly, e.g.
python -m s3a --author=<Author Name>
```
Since every annotation has an associated author, this field must be populated -- and since there is no default author already registered in the app, it cannot make that association. Simply follow the instruction to set a default author:

`python -m s3a --author="username"`

The app will start as expected. As long as the author remains the same, you can start the app in the future without providing an `--author` flag.
## Detailed Feature List

More information about the capabilities of this tool are outlined in the [project wiki](https://gitlab.com/ficsresearch/s3a/-/wikis/home).


## License

This tool is free for personal and commercial use (except the limits imposed by the PyQt5 library). If you publish something based on results obtained through this app, please cite the following papers:

Jessurun, N., Paradis, O., Roberts, A., & Asadizanjani, N. (2020). Component Detection and Evaluation Framework (CDEF): A Semantic Annotation Tool. Microscopy and Microanalysis, 1-5. doi:10.1017/S1431927620018243

