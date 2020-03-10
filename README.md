
# Component Detection and Evaluation Framework (CDEF)
A highly adaptable tool for both visualizing and generating semantic annotations for generic images.

Most software solutions for semantic (pixel-level) labeling are designed for low-resolution (<10MB) images with fewer than 10 components of interest. Violating either constraint (e.g. using a high-res image or annotating $\approx$ 1000 components) incur detrimental performance impacts. CDEF is designed to combat both these deficiencies. With images up to 150 MB and 2000 components, the tool remains interactive. However, since the use case is tailored to multiple small regions of interest within an image, performance lags when editing individual components of larger than $\approx$ 1000x1000 pixels.

## Installation
Clone / cd this repository:
`git clone https://gitlab.com/ficsresearch/cdef`
`cd cdef`
Install requirements:
`pip install -r requirements.txt`

## Usage
Simply run CDEF as a module (`python -m cdef`) and the main window should appear:
![](./docs/img/startup.png)

From here, you can add components by creating polygons in the main window. The newly created component will be enlarged in the focused editor, where you can edit the resulting boundary.
![](./docs/img/compCreation.gif)

The color scheme can also be changed as needed:
![](./docs/img/changeColorScheme.gif)

Just like the color scheme, various application settings can be changed through similar menus:
![](./docs/img/propEditors.png)
CDEF also supports batch component editing through either the component table or main image. To my knowledge, this is a unique feature compared to other annotation tools:
![](./docs/img/batchEdit.gif)

You can also filter which components you want to show on the main image:
![](./docs/img/tableFilter.gif)

Most importantly, you can create your own algorithms for generating segmentation boundaries. These will be available within the GUI, along with any parameters you specify as editable within the parameter window:
![](./docs/img/algEditor.png)
![](./docs/img/liveAlgEdit.gif)





## License

This tool is free for personal and commercial use, provided you cite the following papers:
- sdf
- dsaa


