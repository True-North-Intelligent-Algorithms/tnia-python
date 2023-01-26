---
layout: basic
---

## Indexing Hints

## Get the location of the max of a 3D array

```
np.unravel_index(a.argmax(), a.shape)
```

## Make an index a variable trick

Sometimes it is useful to make an index into a variable to re-use and keep track of it.  This can be done with the ```np.s_``` function.  Below is an example.  

```  
x=100
y=100
z=10
size=128

ind = np.s_[z, y:y+size, x:x+size]
roi = im[ind]
```

## Efficient label filtering 

Sometimes it is useful to filter labels based on some criteria (such as size or circularity), however going through labels and zeroing them on the original array is inefficient.  Instead we can use the ```skimage.util.map_array``` function.  

```
label_indexes = [2, 4, 5, 11]
new_values = [0,0,0,0]
label_image_filtered=util.map_array(label_image, label_indexes, new_values)
```

