# PySOFI
PySOFI is a Python package for SOFI analysis.

You can find a collection of examples under Notebooks folder presented in jupyter notebooks.

### 1. Clone the PySOFI repository
`git clone https://github.com/xiyuyi-at-LLNL/pysofi.git`

### 2. Pip install
if you would like to avoid interfacing with the source code, you can also install this package using the following command from a terminal or the Anaconda prompt:

`pip install pysofi`
### 3. Configuration
PySOFI requires installation of [Anaconda](https://docs.anaconda.com/anaconda/install/).

### 3.1. Create the pysofi virtual environment from the provided .yml file.
Before running PySOFI, create the environment by running the following code in Anaconda prompt or the mac terminal:

`conda env create -f env_MacOS_Majave.yml` (tested for MacOS Majave 10.14.3)

or

`conda env create -f env_Win10.yml` (tested for Windows 10)

or

`conda env create -f env_linux.yml` (tested for Linux kernal version of 3.10.0-1160.45.1.1chaos.ch6.x86_64)


Then activate the environment with:

`conda activate pysofi`

### 3.2. ccreate the pysofi virtual environment manually
First create a conda virtual environment:

`conda create --name pysofi`

Second, activate the pysofi virutal environment using the following command:

`conda activate pysofi`

Next, install basic packages either through conda or through pip. The basic packages include: numpy, scipy, bokeh, opencv-python, scikit-image, scikit-learn, jupyter notebook, ddt, pillow, tifffile, etc.

Due to system variations, other packages may still be missing. This can be identified by trying to perform the unit tests (see section 5 below) or executing the Jupyter notebooks (after step 3.3 shown below) to identify missing packages for additional manual installation.


### 3.3. Add the pysofi kernel of the pysofi virtual environment to Jupyter Notebook
Before running notebook files, set up the environment of Ipython kernel with the following code:

`ipython kernel install --user --name=pysofi`

After starting jupyter notebook, switch kernel to current environment by clicking "Kernel -> Change kernel -> pysofi" from the jupyter notebook dropdown menu.

The notebooks offers 2 options to either import the package from source code, or from pip install. The examples are indicated in the notebook files. If you would like to use the package directly from source, please download this repository, and configure the path as shown in the notebooks.


### 4. Documentation
The documentation for this repository is currently under construction (under `./docs`) using Sphinx, which is also simultaneously being updated on the [online documentation](https://xiyuyi-at-llnl.github.io/pysofi/build/html/index.html) page. We welcome community contribution to the project! 

### 5. Unit tests:
If you have cloned the repository, made modifications to the existing modules and would like to test the codes and make sure the modifications do not cuase break-down of the current fuction, you can go to the pysofi data where all the *.py are enclosed, then type the following command for unit test:

`python -m unittest discover -s .`

### 6. Get involved as a developer
Please refer to our [documentation page about contributing to PySOFI](https://xiyuyi-at-llnl.github.io/pysofi/build/html/about.html#contributing) for details.
If you have further inquiries, please email Xiyu Yi (yi10@llnl.gov) or Yuting Miao (ytmiao@ucla.edu).

### 7. Notice
The work performed by Xiyu Yi is supported under the auspices of the U.S. Department of Energy by Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344. Release number: LLNL-CODE-816626. Please refer to NOTICE.md file for details. The work performed by Yuting Miao is supported by UCLA.


### 8. Complete collection of demo datasetes on figshare
https://figshare.com/s/47d97a2df930380c96bb
