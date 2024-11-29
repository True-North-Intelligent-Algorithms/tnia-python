import dask.array as da
from tnia.deconvolution.richardson_lucy import richardson_lucy_cp
import numpy as np
import cupy as cp
from multiprocessing import Queue

bytes_per_gb = 1024 * 1024 * 1024

def gpu_mem():
    """ get the amount of gpu memory in bytes.  Note this is the total amount of GPU memory on the device.  Not the amount of free memory. 

    Returns:
        int: GPU memory in bytes 
    """
    num_devices = cp.cuda.runtime.getDeviceCount()

    mem_size = []
    
    for i in range(num_devices):
        cp.cuda.Device(i).use()
        mem_info = cp.cuda.runtime.memGetInfo()
        mem_size.append(mem_info[0])
    
    return min(mem_size)

def rl_mem_footprint(img, psf, depth=(0, 0, 0)):
    """ Gets the approximate amount of memory the RL algorithm will use for given image size, psf size and depth.
    The depth is the amount of overlap between chunks that dask will use. The default is 0,0,0 which means no overlap.  The overlap is needed to avoid artifacts at the chunk boundaries.  
    The larger the depth the more memory is needed so ideally the depth should be chosen to be as small as possible while avoiding artifacts.  
    The depth is in pixels.

    Note the memory use returned is approximate.  Due to internal memory use and extra padding used by the FFTs the exact memory use for OpenCL based RL is difficult to compute.  
    Normally the RL algorithm uses 7 buffers, but this function will use a factor of 9, to allow for a factor of safety. 

    Args:
        img (numpy.ndarray): image to be deconvolved
        psf (numpy.ndarray): point spread function
        depth (tuple, optional): Overlap between dask chunks. Defaults to (0, 0, 0).

    Returns:
        int: memory footprint of RL

    Author:
        Dimitris Nicoloutsopoulos
    """
    img_size = np.array(img.shape)
    psf_size = np.array(psf.shape)
    depth_size = np.array(depth)

    total_size = np.prod(img_size + psf_size / 2 + depth_size)

    img_bytes = total_size * 4  # float32 needs 4 bytes
    img_bytes = img_bytes * 11  # the RL deconv algo needs 9 copies but we use 11 for a factor of safety
    return np.ceil(img_bytes)

def chunk_factor(img, psf, depth, mem_to_use=-1):
    """    
    If chunks are needed, it is desirable that the 3D image will be chunked along x and y only because
    the psf is usually elongated along z and chunking it on z could create artifacts.
    This function, will be chunking the image in such a manner that 4 or 16 or 64 etc
    chunks will be used. That means that if we split x by 2 then y will also split y by 2 which
    will result in 4 chunks with the same aspect ratio (on xy) as the original image.
    Similarly, to get 16 chunks (if needed) we will be splitting both x and y by 4
    
    No check is done to determine if the image size is divisible by the chunk size.  
    Any downstream code will need to handle this properly. 

    Args:
        img (numpy.ndarray): image to be deconvolved
        psf (numpy.ndarray): point spread function
        depth (tuple): Overlap between dask chunks.
        mem_to_use (int, optional): Amount of GPU memory to use in GB.  If -1 then use all available memory.  Defaults to -1.

    Returns:
        int: chunk factor

    Author:
        Dimitris Nicoloutsopoulos
        
    """
    if mem_to_use == -1:
        gpu_bytes = gpu_mem()
    else:
        gpu_bytes = mem_to_use*bytes_per_gb
    
    img_bytes = rl_mem_footprint(img, psf, depth)

    cf = 1
    if gpu_bytes <= img_bytes:
        # we want to find an integer k such that:
        # img_bytes / 4^k <= gpu_bytes

        # inflate by 1 byte so that if img_bytes==gpu_bytes then chunks will be produced.
        img_bytes = img_bytes + 1

        k = np.ceil(np.emath.logn(4, img_bytes/gpu_bytes))

        # 4^k is the number of chunks.
        # Now find out how much x and y must be split-by take the square root
        cf = np.sqrt(4 ** k)
    return cf

def richardson_lucy_dask_cp(img, psf, numiterations, non_circulant=True, overlap=10, mem_to_use=-1, debug=False, platform=0, num_devices=1):
    """ perform Richardson-Lucy using dask.  
    
    If there are multiple devices compute() will be called with num_workers=num_devices and the devices ids (assumed to be 0 to num_devices-1) will be put 
    in a Queue and each dask task will pop a device id from the queue.  When the task is complete the device id will be put back in the queue.  
    This will allow multiple GPUs to be used without conflicts. 

    Args:
        img (numpy.ndarray): image to be deconvolved
        psf (numpy.ndarray): point spread function
        numiterations (int): number of iterations 
        non_circulant (bool, optional): If True use non-circulant Richardson Lucy. Defaults to True.
        overlap (int, optional): Overlap between blocks. Defaults to 10.
        mem_to_use (int, optional): GPU memory to use in GB.  If -1 use full GPU memory, otherwise limit GPU memory to mem_to_use. Defaults to -1.
        debug (bool, optional): If True print debug information. Defaults to False.
        platform (int, optional): OpenCL platform to use. Defaults to 0.
        num_devices (int, optional): Number of GPUs to use. Defaults to 1.
    Returns:
        numpy.ndarray: deconvolved image 
    """
    if debug:
        print('richardson_lucy_dask_cp')
        print()
        print('image size',img.shape)
        print('psf size', psf.shape)

    gpu_mem_ = gpu_mem()/bytes_per_gb
    
    if debug:
        print('gpu mem is ', gpu_mem_)

    if len(img.shape) == 3:
        depth = (0, overlap, overlap)
    else:
        depth = (overlap, overlap)

    rl_mem_ = rl_mem_footprint(img, psf, depth=depth)/bytes_per_gb
    
    if debug:
        print('rl mem is ', rl_mem_)

    k = chunk_factor(img, psf, depth=depth, mem_to_use=mem_to_use)
    
    if debug:
        print('chunk factor is ', k)

    if img.shape[-2] % k != 0:
        y_chunk_size = img.shape[-2] // k + 1
    else:
        y_chunk_size = img.shape[-2] // k

    if img.shape[-1] % k != 0:
        x_chunk_size = img.shape[-1] // k + 1
    else:
        x_chunk_size = img.shape[-1] // k

    if len(img.shape) == 3:
        chunk_size = (img.shape[0], y_chunk_size, x_chunk_size)
        
        if debug:
            print('chunk size is',chunk_size)

        dimg = da.from_array(img,chunks=(img.shape[0], y_chunk_size, x_chunk_size))
    else:
        chunk_size = (y_chunk_size, x_chunk_size)
        
        if debug:
            print('chunk size is',chunk_size)

        dimg = da.from_array(img,chunks=(y_chunk_size, x_chunk_size))

    queue = Queue()

    for i in range(num_devices):
        queue.put(i)

    import traceback

    def rl_dask_task(img, psf, numiterations, block_info=None, block_id=None, thread_id=None):

        print('start dask task')
        
        try:
            # check if any image dimension is 0 for arbitraryh length array
            print('check dimensions')
            
            if np.any(np.array(img.shape) == 0):
                print('image dimension is 0')
                return None
            if block_id is None or block_id == []:
                return None
            
            print()
            device_num=queue.get()
            
            if debug:
                print('start rlnc')
                print('gpu num is', device_num)
                print('block id is', block_id)
                print('block info is', block_info)
            
            cp.cuda.Device(device_num).use()
            result=richardson_lucy_cp(img, psf, numiterations, non_circulant)
            
            return result
        except Exception as e:
            traceback.print_exc()
            if debug:
                print()
                print("EXCEPTION",e)
            pass
        finally:
            if debug:
                print('putting gpu num back', device_num)
            queue.put(device_num)
    
    import time

    start_time = time.time()

    if len(img.shape) == 3:
        depth = {0:0, 1:overlap, 2:overlap}
    else:
        depth = {0:overlap, 1:overlap}

    out = dimg.map_overlap(rl_dask_task, depth=depth, dtype=np.float32, psf=psf, numiterations=numiterations)
    out_img = out.compute(num_workers=num_devices)
    end_time = time.time()
    execution_time = end_time - start_time
    if debug:
        print(f"Execution time of rl dask multi gpu: {execution_time} seconds")
    return out_img 





