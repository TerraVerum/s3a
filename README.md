
# Component Detection and Evaluation Framework (CDEF)
A highly adaptable tool for both visualizing and generating semantic annotations for generic images.

Most software solutions for semantic (pixel-level) labeling are designed for low-resolution (<10MB) images with fewer than 10 components of interest. Violating either constraint (e.g. using a high-res image or annotating $\approx$ 1000 components) incur detrimental performance impacts. CDEF is designed to combat both these deficiencies. With images up to 150 MB and 2000 components, the tool remains interactive. However, since the use case is tailored to multiple small regions of interest within an image, performance lags when editing individual components of larger than $\approx$ 1000x1000 pixels.

___

A more detailed overview can be found in the project wiki [here](https://gitlab.com/ficsresearch/cdef/-/wikis/docs/user's-guide).

___

## Installation

Clone / cd this repository:
```bash
git clone https://gitlab.com/ficsresearch/cdef
cd cdef
```
Install requirements:
```bash
pip install -r requirements.txt
```
Optionally, the repository can be installed as a pip module:
```bash
pip install .
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
## Command-Line Arguments
You can specify more than just the author at startup. If you want a certain image to be opened on launch, specify the filepath with the `--image` switch:

```bash
python -m cdef --image="./Images/circuitBoard.png"
```

When CDEF opens, the requested image will be loaded. The following command-line arguments are accepted:
- `--author`: Username of the person currently making annotations. This will appear next to each component created during the session.
- `--image`: Filename of the starting image
- `--annotations`: Annotation file to load in on startup
- `--profile`: Name of a saved profile. If additional flags are provided, they will override settings from this profile.
- `--layout`: Name of a saved Layout
- `--"<Editor Name>"`: Name of the saved parameter editor state. For a list of allowed editors, see the user guide.

## Usage
Simply run CDEF as a module (as explained in [Command-Line Arguments](#Command-Line Arguments)) and the main window should appear:

![](./docs/img/readme/startup.png)

From here, you can add components by creating polygons in the main window. The newly created component will be enlarged in the focused editor, where you can edit the resulting boundary.

![](./docs/img/readme/compCreation.gif)

The color scheme can also be changed as needed:

![](./docs/img/readme/changeColorScheme.gif)

Just like the color scheme, various application settings can be changed through similar menus:

![](./docs/img/readme/propEditors.png)

CDEF also supports batch component editing through either the component table or main image. To my knowledge, this is a unique feature compared to other annotation tools:

![](./docs/img/readme/batchEdit.gif)

You can also filter which components you want to show on the main image:

![](./docs/img/readme/tableFilter.gif)

Most importantly, you can create your own algorithms for generating segmentation boundaries. These will be available within the GUI, along with any parameters you specify as editable within the parameter window:

![](./docs/img/readme/algEditor.png)

![](./docs/img/readme/liveAlgEdit.gif)


## License

This tool is free for personal and commercial use, provided you cite the following papers:

{{Awaiting publication}}

