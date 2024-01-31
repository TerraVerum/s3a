<div align="center">
<img src="https://gitlab.com/s3a/s3a/-/wikis/imgs/home/s3alogo-square.svg" width="150px"/>
<h1>Semi-Supervised Semantic Annotator (S3A)</h1>

[![pipeline status](https://gitlab.com/s3a/s3a/badges/development/pipeline.svg)](https://gitlab.com/s3a/s3a/-/commits/development)
[![coverage report](https://gitlab.com/s3a/s3a/badges/development/coverage.svg)](https://gitlab.com/s3a/s3a/-/commits/development)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/ntjess/s3a.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/ntjess/s3a/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/ntjess/s3a.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/ntjess/s3a/context:python)
[![SonarCloud](https://img.shields.io/static/v1?label=Scanned%20On&message=SonarCloud&color=orange)](https://sonarcloud.io/summary/new_code?id=s3a_s3a)

<img src="https://gitlab.com/s3a/s3a/-/wikis/s3a-window.jpg" width="75%"/>
</div>

## Description

A highly adaptable tool for both visualizing and generating semantic annotations for generic images. You can also use
the model plugin system to rapidly test and improve computer vision algorithms/models. This latter use-case is the
unintended primary use among some users.

Most software solutions for semantic (pixel-level) labeling are designed for low-resolution (<10MB) images with fewer 
than 10 components of interest. Violating either constraint (e.g. using a high-res image or annotating ~1000 
components) led to detrimental performance impacts when this project was started. S3A is designed to combat both these 
deficiencies. It remains interactive With images up to 150 MB and 2000 components.

A more detailed overview can be found in the project wiki [here](https://gitlab.com/ficsresearch/s3a/-/wikis/docs/user's-guide).

## Install

You can install s3a on Windows or Linux by git cloning and pip installing this repository. The following scripts should work equally well in PowerShell as in bash, presuming git and python are installed.

If you don't already know, you might get a virtual environment setup first to avoid mucking up your user or system space.
```bash
python -m pip install virtualenv
python -m virtualenv venv
./venv/bin/activate # Or on windows likely "./venv/bin/Activate"
```

```bash
git clone https://github.com/TerraVerum/s3a
cd s3a
pip install .
```

## Run

You can run the app with either of the following commands:

```bash
python -m s3a
```

```bash
s3a-gui
```

From here, projects can be created to host groups of related images, or images can be annotated in the default project. Both options are available through the `File` menu.

## Detailed Feature List

More information about the capabilities of this tool are outlined in the [project wiki](https://gitlab.com/s3a/s3a/-/wikis/home).

## <span style="color:red">Please Note</span>
S3A's programmatic API is still largely under development. It still needs refinement to allow for consistent naming schemes, removing vestigial elements, confirming private vs. public-facing elements, and a few other line items. However, the graphical interface should be minimally affected by these alterations.

Thus, while the GUI entry point should be consistently useful, be aware of these developments when using the scripting portion of S3A in great detail.

## License

This tool is free for personal and commercial use (except the limits imposed by the selected Qt binding). If you publish something based on results obtained through this app, please cite the following paper:

Jessurun, N., Paradis, O., Roberts, A., & Asadizanjani, N. (2020). Component Detection and Evaluation Framework (CDEF): A Semantic Annotation Tool. Microscopy and Microanalysis, 1-5. doi:10.1017/S1431927620018243

