# calc_sum_shared.py
import sys
import numpy as np
from multiprocessing import shared_memory
import cellpose
from cellpose import models

if __name__ == "__main__":
    
    name = sys.argv[1]
    shape = (int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    out_name = sys.argv[5]
    
    dtype = np.dtype(sys.argv[6])

    shm = shared_memory.SharedMemory(name=name)
    array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    print('shape in', shape)

    total = np.sum(array)
    print(f"Sum of shared memory array: {total}")
    
    major_number = cellpose.version.split('.')[0]
    print(f"Cellpose version: {cellpose.version} (major number: {major_number})")
    
    if major_number == '3':
        model = models.Cellpose(gpu=True, model_type='cyto2')
    elif major_number == '4':
        model = models.CellposeModel(gpu=True)

    result = model.eval(array, niter=2000)[0]
    
    print("out_name", out_name)
    
    shape_out = [shape[0], shape[1]]

    print("shape_out", shape_out)
    dtype_out = np.uint16
    nbytes = np.prod(shape_out)* np.dtype(dtype_out).itemsize
    print("nbytes", nbytes)
    shm_out = shared_memory.SharedMemory(size=int(nbytes), name=out_name)
    # Create a numpy array backed by shared memory
    shared_array_out = np.ndarray(shape_out, dtype=dtype, buffer=shm_out.buf)

    # Copy the image data to shared memory
    shared_array_out[:] = result[:]

    shm.close()  # Do NOT unlink — only creator does that