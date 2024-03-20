from tnia.io.io_helper import open_stack
from skimage.io import imsave, imread 
from tnia.io.raw import open_arw

dir= 'C:/...'
ext= 'tif'
stack = open_stack(dir,ext,imread)

stack=stack[:,:,:,1]

imsave(dir + '/stack.tif',stack)
