# Tunneler
## Installation
### Prerequisites
#### Software:
- python3: https://www.python.org/downloads/

#### Python packages
- numpy, scipy, shapely, matplotlib, tkinter, and scikit-learn; install e.g. with pip:
```
$ pip3 install numpy scipy shapely matplotlib scikit-learn tk
```

On macOS, you may have to install the proper python3 tkinter in [this way]([url](https://stackoverflow.com/questions/68554782/modulenotfounderror-no-module-named-tkinter-on-macos)):
```
$ brew install python-tk
```

### Plugin installation
Simply copy the plg .py files in the /plg/ subdirectory of your Yasara installation folder, e.g on Mac typically /Applications/YASARA.app/yasara/plg/
Yasara may try to use it's own, reduced python version, which doesn't allow installation of packages. 
To avoid, rename the epy folder in the /yasara/ directory, e.g on Mac on a console type 
`mv /Applications/YASARA.app/yasara/epy /Applications/YASARA.app/yasara/epy_disable`

## Usage
Try the default settings first.
To use MD settings, you need Yasara structure.
If plugin is too slow / laggy, try reducing the ball spacing parameter.
If the tunnels show big patches on the surface and you are only interested in the inner ones, increase the two surface parameters.
