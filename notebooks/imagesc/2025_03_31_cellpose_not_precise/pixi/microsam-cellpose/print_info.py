import numpy
import torch
import micro_sam
import cellpose
import sys

print('Python:', sys.version)
print('numpy is', numpy.__version__)
print('Torch:', torch.__version__, '| GPU:', torch.cuda.is_available(), '| Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
print('micro_sam:', micro_sam.__version__)
print('cellpose:', cellpose.version)
