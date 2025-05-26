## dependencies

### microsam, mobilesam via segment-everything with cellpose latest

(seems like Cellpose was updated on pypi May 5 2025 to be Cellpose 4, so below instructions should grab cp4...)

however since cp4 is pretty new could be a good idea to get it from github (```pip install git+https://www.github.com/mouseland/cellpose.git```) if below instruction don't work. 

```
conda create -n microsam_cellose_sam -c conda-forge python=3.11
conda activate microsam_cellpose
conda install -c conda-forge micro_sam
pip install cellpose
pip install git+https://github.com/True-North-Intelligent-Algorithms/segment-everything.git
```

### microsam, mobilesam via segment-evertyhing with cellpose < 4 (no cellpose SAM)

```
conda create -n microsam_cellose_sam -c conda-forge python=3.11
conda activate microsam_cellpose
conda install -c conda-forge micro_sam
pip install "cellpose<4" 
pip install git+https://github.com/True-North-Intelligent-Algorithms/segment-everything.git
```