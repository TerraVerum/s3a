<div align="center">
<img src="https://gitlab.com/s3a/s3a/-/wikis/imgs/home/s3alogo-square.svg" width="150px"/>
<h1>Semi-Supervised Semantic Annotator (S3A)</h1>

![Pipeline](https://github.com/terraverum/s3a/actions/workflows/pipeline.yaml/badge.svg)

<img src="https://gitlab.com/s3a/s3a/-/wikis/s3a-window.jpg" width="75%"/>
</div>

## Description

A highly adaptable tool built by TerraVerum's founder during his PhD program for both visualizing and generating 
semantic annotations for generic images. You can also use the model plugin system to rapidly test and improve computer
vision algorithms/models.

We're continuing to provide a level of support but, to avoid brand confusion, this is _not_ our intended commercial 
offering. Some concepts apply, but this isn't it.

A more detailed overview can be found in the project wiki [here](https://gitlab.com/s3a/s3a/-/wikis/docs/user's-guide).

## Install

You can install s3a on Windows or Linux by git cloning and pip installing this repository. The following scripts should work equally well in PowerShell as in bash, presuming git and python are installed.

```bash
git clone https://github.com/TerraVerum/s3a
cd s3a
```

If you don't already know, you might get a [virtual environment](https://docs.python.org/3/library/venv.html) setup before you do the following pip install to avoid mucking up your user or system space.

```bash
python -m pip install virtualenv
python -m virtualenv venv
source venv/bin/activate # If on Linux
.\venv\Scripts\activate # if on Windows
```

If you can't activate the virtual environment on Windows, you may need to run this command to enable script execution:

```ps
Set-ExecutionPolicy --Scope CurrentUser Unrestricted
```

With or without a virtual environment, now install S3A via pip:

```bash
pip install .
```

## Test

You can run the unit tests by installing the optional dependencies and then running pytest from inside this repository:

```bash
pip install .[full]
pytest -l ./apptests --cov-report term --cov-report xml --cov=./s3a
```

**Note:** The tests may be slow to start the first time.

## Run

You can run the app with either of the following commands once installed. (If using a virtual environment, it'll only be availible with that environment activated)

```bash
python -m s3a
# OR
s3a-gui
```

From here, projects can be created to host groups of related images, or images can be annotated in the default project. Both options are available through the `File` menu.

**Note:** the latest code-base is not provided as a standalone .exe as we're not finding a great way to keep the .exe 
from spawning virus alerts on Windows 11.

**Note:** the app is very slow to startup the first time.

## Detailed Feature List

More information about the capabilities of this tool are outlined in the [project wiki](https://gitlab.com/s3a/s3a/-/wikis/home).

## <span style="color:red">Please Note</span>
S3A's programmatic API is still largely under development. It still needs refinement to allow for consistent naming schemes, removing vestigial elements, confirming private vs. public-facing elements, and a few other line items. However, the graphical interface should be minimally affected by these alterations.

Thus, while the GUI entry point should be consistently useful, be aware of these developments when using the scripting portion of S3A in great detail.

## License

This tool is free for personal and commercial use (except the limits imposed by the selected Qt binding). If you publish something based on results obtained through this app, please cite the following paper:

Jessurun, N., Paradis, O., Roberts, A., & Asadizanjani, N. (2020). Component Detection and Evaluation Framework (CDEF): A Semantic Annotation Tool. Microscopy and Microanalysis, 1-5. doi:10.1017/S1431927620018243
