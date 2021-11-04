from skimage.io import imread, imshow
import matplotlib.pyplot as plt
import os

input_image = imread("D:\\images\\from broad\\BBBC032_v1_dataset/BMP4blastocystC3-cropped_resampled_8bit_2D.tif")

# plot input image
fig, ax = plt.subplots(figsize=(12,9))
ax.imshow(input_image)
ax.set_title('Intput Image')

report_dir = '../../docs/tests/'
if os.path.exists(report_dir)==False:
    os.makedirs(report_dir)

fig.savefig(report_dir+'test.png')

markdown='## Test \n\n'
markdown+='This is a test \n\n'
markdown+='Test image below \n\n'
markdown+='![Test image](test.png) \n\n'

outfile=open(report_dir+'test.md', "w")
outfile.write(markdown)
outfile.close()
