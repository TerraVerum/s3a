![](./docs/img/readme/cdefLogo.svg)

# Component Detection and Evaluation Framework (CDEF)

A highly adaptable tool for both visualizing and generating semantic annotations for generic images.

Most software solutions for semantic (pixel-level) labeling are designed for low-resolution (<10MB) images with fewer than 10 components of interest. Violating either constraint (e.g. using a high-res image or annotating $\approx$ 1000 components) incur detrimental performance impacts. CDEF is designed to combat both these deficiencies. With images up to 150 MB and 2000 components, the tool remains interactive. However, since the use case is tailored to multiple small regions of interest within an image, performance lags when editing individual components of larger than $\approx$ 1000x1000 pixels.

___

A more detailed overview can be found in the project wiki [here](https://gitlab.com/ficsresearch/cdef/-/wikis/docs/user's-guide).

___

## Installation

Clone and install the dependent `imageprocessing` repository:

```bash
git clone https://gitlab.com/ficsresearch/imageprocessing
pip install -e ./imageprocessing
```

Next, clone and install `cdef`:

```bash
git clone https://gitlab.com/ficsresearch/cdef
pip install -e ./cdef
```

## Running the App
Running the app is as easy as calling `cdef` as a module:
`python -m cdef`

However, if this is the first time you are starting CDEF, you will run into the following error message:
```
No author name provided and no default author exists. Exiting.
To start without error, provide an author name explicitly, e.g.
python -m cdef --author=<Author Name>
```
Since every annotation has an associated author, this field must be populated -- and since there is no default author already registered in the app, it cannot make that association. Simply follow the instruction to set a default author:

`python -m cdef --author="username"`

The app will start as expected. As long as the author remains the same, you can start the app in the future without providing an `--author` flag.
## Detailed Feature List

More information about the capabilities of this tool are outlined in the [project wiki](https://gitlab.com/ficsresearch/cdef/-/wikis/home).


## License

This tool is free for personal and commercial use, provided you cite the following papers:

{{Awaiting publication}}

