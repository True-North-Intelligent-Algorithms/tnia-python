"""
Created on Wed Mar 17 09:22:33 2021

@author: bnorthan
"""

import javabridge
import bioformats as bf
import numpy as np

def start_jvm():
    javabridge.start_vm(class_path=bf.JARS)

def kill_jvm():
    javabridge.kill_vm()
 
def load_volume(filename, c=0, series=0):
    """ load a volume (all x,y,z) from a bioformats 5D (x,y,z,c,s) series.  s is series number

    Args:
        filename (string): name of bioformats supported file
        c (int, optional): channel number. Defaults to 0.
        series (int, optional): series number. Defaults to 0.

    Returns:
        [3D numpy Array]: xyz volume at channel c and series number s 
    """
    
    meta=bf.get_omexml_metadata(filename)
    o=bf.OMEXML(meta)

    size_x=o.image(2).Pixels.get_PhysicalSizeX()
    size_y=o.image(2).Pixels.get_PhysicalSizeY()
    size_z=o.image(2).Pixels.get_PhysicalSizeZ()
    
    nz=o.image(series).Pixels.SizeZ   
    
    img=bf.load_image(filename, z=0,c=c,series=series,rescale=False)
    img=img[np.newaxis,...]
    
    for cz in range(1,nz):
        temp=bf.load_image(filename, z=cz, c=c, series=series, rescale=False)
        img=np.vstack((img, temp[np.newaxis,...]))
        
    return img, size_x, size_y, size_z

   
def load_channel(filename, nz,c):    
    
    img=bf.load_image(filename, z=0,c=c,rescale=False)
    img=img[np.newaxis,...]
    
    for cz in range(1,nz):
        print() 
        print() 
        print() 
        print('read slice',cz) 
        temp=bf.load_image(filename, z=cz, c=c, rescale=False)
        img=np.vstack((img, temp[np.newaxis,...]))
        
    return img

def load_plane(filename,z,c):    
    return bf.load_image(filename, z=z,c=c,rescale=False)

def save_czyx(file_name, img, channel_names=None):
    """ save a 4d image (c,z,y,x) order

    Args:
        filename ([type]): [description]
        img ([type]): [description]
    """
    if channel_names == None:
        channel_names = ['a','b','c','d']

    nc = img.shape[0]
    nz = img.shape[1]
    ny = img.shape[2]
    nx = img.shape[3]

    for z in range(nz):
        for c in range(nc):
            bf.write_image(file_name,img[c,z,:,:],'uint16',c=c,z=z,t=0,size_t=1,size_c=nc, size_z=nz, channel_names=channel_names)

def save_zcyx(file_name, img, channel_names=None):
    """ save a 4d image (c,z,y,x) order

    Args:
        filename ([type]): [description]
        img ([type]): [description]
    """
    if channel_names == None:
        channel_names = ['a','b','c','d']

    nz = img.shape[0]
    nc = img.shape[1]
    ny = img.shape[2]
    nx = img.shape[3]

    for z in range(nz):
        for c in range(nc):
            bf.write_image(file_name,img[z,c,:,:],'uint16',c=c,z=z,t=0,size_t=1,size_c=nc, size_z=nz, channel_names=channel_names)