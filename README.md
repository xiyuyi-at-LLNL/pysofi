# PySOFI
PySOFI is a Python package for SOFI analysis.

You can find a collection of examples under Notebooks folder presented in jupyter notebooks.

### Configuration
PySOFI requires installation of [Anaconda](https://docs.anaconda.com/anaconda/install/).

Before running PySOFI, create the environment by running the following code in Anaconda prompt or the mac terminal:

`conda env create -f pysofi.yml`

Then activate the environment with:

`conda activate pysofi`

Before running notebook files, set up the environment of Ipython kernel with the following code:

`ipython kernel install --user --name=pysofi`

After starting jupyter notebook, switch kernel to current environment by clicking "Kernel -> Change kernel -> pysofi" from the jupyter notebook dropdown menu.

The notebooks offers 2 options to either import the package from source code, or from pip install. The examples are indicated in the notebook files. If you would like to use the package directly from source, please download this repository, and configure the path as shown in the notebooks.

if you would like to avoid interfacing with the source code, you can also install this package using hte following command from the terminal:

`pip install pysofi`

### Documentation
The documentation for this repository is currently under construction (under `./docs`) using Sphinx, which is also simultaneously being updated on the [online documentation](https://xiyuyi-at-llnl.github.io/pysofi/build/html/index.html) page. We welcome community contribution to the project! 

### To perform tests for your own updates:
If you have clone the repository, made modifications to the existing modules and would like to make sure the modifications do not cuase break-down of the current fuction, you can go to the pysofi data where all the *.py are enclosed, then type the following command for unit test:

`python -m unittest discover -s .`

### Get involved as a developer
Please refer to our [documentation page about contributing to PySOFI](https://xiyuyi-at-llnl.github.io/pysofi/build/html/about.html#contributing) for details.
If you have further inquiries, please email Xiyu Yi (yi10@llnl.gov) or Yuting Miao (ytmiao@ucla.edu).

### Notice
The work performed by Xiyu Yi is supported under the auspices of the U.S. Department of Energy by Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344. Release number: LLNL-CODE-816626. Please refer to NOTICE.md file for details. The work performed by Yuting Miao is supported by UCLA.



### Extra demo datasetes on figshare
https://figshare.com/s/47d97a2df930380c96bb
