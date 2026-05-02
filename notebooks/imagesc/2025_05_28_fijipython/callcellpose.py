#@ ImageJ ij
#@ ImagePlus imp

import cellpose
from cellpose import models
print(cellpose.version)

model = cellpose.models.CellposeModel(gpu=True, model_type='cyto3')

print('processing img',imp.getTitle(), 'with cellpose', cellpose.version)

img_py=ij.py.from_java(imp)
result = model.eval(img_py)

result_imp = ij.py.to_imageplus(result[0])
result_imp.setTitle("result")

ij.ui().show(result_imp)

ij.IJ.setThreshold(result_imp, 1, 100000)
ij.py.run_plugin("Convert to Mask", imp=result_imp);

ij.py.run_plugin("Create Selection", imp=result_imp);
ij.py.run_plugin("Add to Manager"); 

ij.IJ.selectWindow(imp.getTitle()); 
ij.py.run_plugin("Show Overlay", imp=imp);

