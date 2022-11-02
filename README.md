<div align="center">
 
![logo](https://github.com/bao18/open_earth_map/blob/main/pics/openearthmap.png)
[![Generic badge](https://img.shields.io/badge/License-MIT-<COLOR>.svg?style=for-the-badge)](https://github.com/bao18/open_earth_map/blob/main/LICENSE) 
[![PyTorch - Version](https://img.shields.io/badge/PYTORCH-1.12+-red?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/previous-versions/) 
[![Python - Version](https://img.shields.io/badge/PYTHON-3.7+-red?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/) 


</div>

<!-- 
# OpenEarthMap
Quick start in OpenEarthMap  -->
The main features of this library are:

 - High-level API (only two lines to create a neural network)
 - Three models architectures for multi-class segmentation (including the popular U-Net)
 - Popular metrics and losses for training routines

### Compile full OpenEarthMap dataset
To compile the full OpenEarthMap, the xBD dataset is needed. Please download both datasets, [OpenEarthMap](https://zenodo.org/record/7223446#.Y2Jj1OzP2Ak) and [xBD](https://xview2.org/dataset). Then, run the following command:
```python
python data/compile_xbd.py \
    --path_to_OpenEarthMap "folder where OpenEarthMap is located" \
    --path_to_xBD "folder where xBD is located"
```

<!-- ### Example <a name="examples"></a> -->
### Example
This example shows the application for multi-class semantic segmentation using a small version of the OpenEarthMap dataset. Please, follow the  [notebook](https://github.com/bao18/open_earth_map/blob/main/Demo.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bao18/open_earth_map/blob/main/Demo.ipynb)

