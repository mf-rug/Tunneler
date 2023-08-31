# Tunneler
## Installation
### Prerequisites
#### Software:
- python3: https://www.python.org/downloads/
- anaconda: https://www.anaconda.com/download or miniconda: https://docs.conda.io/en/latest/miniconda.html
#### Python packages
- numpy and scipy, install e.g. with pip:
```
$ pip3 install numpy scipy
```
- PCL: I recommend installing via a conda environment
```
$ conda create pcl_env
$ conda activate pcl_env
(pcl_env) $ conda install -c sirokujira pcl --channel conda-forge
(pcl_env) $ conda install -c sirokujira python-pcl --channel conda-forge
```

### Plugin installation
Simply copy the plg .py file in the /plg/ subdirectory of your Yasara installation folder, e.g on Mac typically /Applications/YASARA.app/yasara/plg/

## Usage
