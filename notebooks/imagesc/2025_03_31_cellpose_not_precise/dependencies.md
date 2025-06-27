conda create -n microsam_cellpose -c conda-forge python=3.11
conda activate microsam_cellpose
conda install microsam
conda install cellpose
conda install mondai
python -c "import numpy; print(numpy.__version__)"
python -c "import torch; print('Torch:', torch.__version__, '| GPU:', torch.cuda.is_available(), '| Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"


