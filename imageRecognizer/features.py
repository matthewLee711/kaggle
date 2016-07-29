import numpy as np
import Image

'''
Extracts features from an input image
1. Converts an image into a 1D vector.
2. 
'''
img = Image.open('orig.png').convert('RGBA')
arr = np.array(img)

# record the original shape
shape = arr.shape

# make a 1-dimensional view of arr
flat_arr = arr.ravel()

# convert it to a matrix
vector = np.matrix(flat_arr)

# do something to the vector
vector[:,::10] = 128

# reform a numpy array of the original shape
arr2 = np.asarray(vector).reshape(shape)

# make a PIL image
img2 = Image.fromarray(arr2, 'RGBA')
img2.show()
