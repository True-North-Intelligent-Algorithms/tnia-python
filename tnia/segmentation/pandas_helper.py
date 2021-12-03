
import pandas as pd
from skimage.measure import regionprops

def get_df_centroid_intensity_volume_3D(labels, img, sx=1, sy=1, sz=1):
    stats=pd.DataFrame(columns=['x','y','z','intensity','volume'])
    
    object_list=regionprops(labels,img)

    for o in object_list:
        c=o.centroid
        v=o.area
        i=v*o.mean_intensity

        stats.loc[len(stats)]=[c[2]*sx,c[1]*sy,c[0]*sz,i,v*sx*sy*sz]

    return stats
 