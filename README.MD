# performs sorting and grouping operations on multidimensional NumPy arrays using Cython and hash-based algorithms.


## pip install cythonnestednumpy

### Tested against Windows / Python 3.11 / Anaconda


## Cython (and a C/C++ compiler) must be installed



```python
import numpy as np
from cythonnestednumpy import HashSort

img1 = np.full([900,1800,3],255,dtype=np.uint8)
img2 = np.full([900,1800,3],255,dtype=np.uint8)
img2[...,0]=0
img3 = np.full([900,1800,3],255,dtype=np.uint8)
img3[...,2]=1
a=np.concatenate([img1,img2,img3])
cyne=HashSort(a,unordered=True)
# 1st column: absolut index (using a.flatten() or a.ravel())
# 2nd - n column: dimension (the more dimensions your array has, the more columns will show up)
# 3rd column: The index in cyne.iterray
# 4th column: 1 is for the first item (unique) found. 0 means that there has been found the same value before.
# 5th column: How many matches
# 6th column: Hashcode


cyne.generate_hash_array(last_dim=None)
resultdata=cyne.sort_by_absolut_index(ascending=True)
# Out[3]:
# array([[                   0,                    0,                    1,                    1,  2497830064280488930],
#        [                  30,                    1,                    0,                    2,  2497830064280488930],
#        [                  60,                    2,                    0,                    3,  2497830064280488930],
#        [                  90,                    3,                    0,                    4,  2497830064280488930],
# ...
#        [                 780,                   26,                    0,                    7, -5024405870974420794],
#        [                 810,                   27,                    0,                    8, -5024405870974420794],
#        [                 840,                   28,                    0,                    9, -5024405870974420794],
#        [                 870,                   29,                    0,                   10, -5024405870974420794]], dtype=int64)
cyne.sort_by_absolut_index(ascending=False)

# Out[4]:
# array([[                 870,                   29,                    0,                   10, -5024405870974420794],
#        [                 840,                   28,                    0,                    9, -5024405870974420794],
#        [                 810,                   27,                    0,                    8, -5024405870974420794],
#        [                 780,                   26,                    0,                    7, -5024405870974420794],
#        [                 750,                   25,                    0,                    6, -5024405870974420794],
#        [                 720,                   24,                    0,                    5, -5024405870974420794],
# ...
#        [                 240,                    8,                    0,                    9,  2497830064280488930],
#        [                 210,                    7,                    0,                    8,  2497830064280488930],
#        [                 180,                    6,                    0,                    7,  2497830064280488930],
#        [                 150,                    5,                    0,                    6,  2497830064280488930],
#        [                 120,                    4,                    0,                    5,  2497830064280488930],
#        [                  90,                    3,                    0,                    4,  2497830064280488930],
#        [                  60,                    2,                    0,                    3,  2497830064280488930],
#        [                  30,                    1,                    0,                    2,  2497830064280488930],
#        [                   0,                    0,                    1,                    1,  2497830064280488930]], dtype=int64)


cyne.get_unique_dims_data()
# Out[3]:
# [array([255, 255, 255], dtype=uint8),
#  array([255, 255, 255], dtype=uint8),
#  array([255, 255, 255], dtype=uint8)]

cyne.get_unique_dims_data(start_dim=2,end_dim=-1)
#          [255, 255, 255],
#          [255, 255, 255],
#          [255, 255, 255],
#          [255, 255, 255],
#          [255, 255, 255]],
#
#         [[255, 255, 255],
#          [255, 255, 255],
#           ...
#          [  0, 255, 255]],
#
#         [[  0, 255, 255],
#          [  0, 255, 255],
#          [  0, 255, 255],
#          [  0, 255, 255],
#          [  0, 255, 255],
#          [  0, 255, 255],
#          [  0, 255, 255],
#          [  0, 255, 255],
#          [  0, 255, 255],
#          [  0, 255, 255]],

cyne.get_all_values(start_dim=0,end_dim=-1)
#         ...,
#         [255, 255, 255],
#         [255, 255, 255],
#         [255, 255, 255],
#         [255, 255, 255],
#         [255, 255, 255],
#         [255, 255, 255],
#         [255, 255, 255],
#         [255, 255, 255],
#         [255, 255, 255]],
#        ...,
#        [[255, 255,   1],
#         [255, 255,   1],
#         [255, 255,   1],
#         [255, 255,   1],
#         [255, 255,   1],
#         [255, 255,   1],
#         [255, 255,   1],
#         [255, 255,   1],
#         [255, 255,   1],
#         [255, 255,   1]],
#          ...,
#allva=cyne.get_unique_dims_values()
cyne.group_equal_values()
#         [255, 255,   1]],
#        [[255, 255,   1],
#         [255, 255,   1],
#         [255, 255,   1],

#         [  0, 255, 255],
#         [  0, 255, 255],
#         ...

#         [255, 255, 255],
#         [255, 255, 255],
#         [255, 255, 255],
#         [255, 255, 255],
#         [255, 255, 255],
#         [255, 255, 255]],

groupedvalues=cyne.group_equal_values()
byqty=cyne.sort_by_quantity(ascending=False)
cyne.sort_by_hash(ascending=False)
# Out[3]:
# array([[                 270,                    9,                    0,                   10,  2497830064280488930],
#        [                 240,                    8,                    0,                    9,  2497830064280488930],
#        [                 210,                    7,                    0,                    8,  2497830064280488930],
#        [                 180,                    6,                    0,                    7,  2497830064280488930],
# ...
#        [                 750,                   25,                    0,                    6, -5024405870974420794],
#        [                 720,                   24,                    0,                    5, -5024405870974420794],
#        [                 690,                   23,                    0,                    4, -5024405870974420794],
#        [                 660,                   22,                    0,                    3, -5024405870974420794],
#        [                 630,                   21,                    0,                    2, -5024405870974420794],
#        [                 600,                   20,                    1,                    1, -5024405870974420794]], dtype=int64)

class HashSort(builtins.object)
 |  HashSort(a, unordered=True)
 |  
 |  The HashSort class is designed to perform sorting and grouping operations on multi-dimensional NumPy arrays
 |  using a hash-based algorithm. It utilizes the xxhash https://xxhash.com/ function (Cython! Not Python!) for efficient hash computation.
 |  
 |  Parameters:
 |  - a (numpy.ndarray): The input multi-dimensional NumPy array.
 |  - unordered (bool): If True, will create the index array with multi processing
 |  
 |  Methods:
 |  - generate_hash_array(last_dim=None): Generates a hash array based on the provided array and optional last_dim.
 |  - sort_by_absolut_index(ascending=True): Sorts the hash array by absolute index in ascending or descending order.
 |  - get_unique_dims_data(start_dim=0, end_dim=-1): Returns unique dimensions data based on hash array.
 |  - get_all_values(start_dim=0, end_dim=-1): Returns all values based on hash array and specified dimensions.
 |  - group_equal_values(start_dim=0, end_dim=-1): Groups equal values based on hash array and specified dimensions.
 |  - sort_by_hash(ascending=False): Sorts the hash array by hash values in ascending or descending order.
 |  - sort_by_quantity(ascending=False): Sorts the hash array by quantity values in ascending or descending order.
 |  
 |  Methods defined here:
 |  
 |  __init__(self, a, unordered=True)
 |      Initializes a new instance of the HashSort class.
 |      
 |      Parameters:
 |      - a (numpy.ndarray): The input multi-dimensional NumPy array.
 |      - unordered (bool): If True, will create the index array with multi processing
 |  
 |  generate_hash_array(self, last_dim=None)
 |      Generates a hash array based on the provided array and optional last_dim.
 |      
 |      Parameters:
 |      - last_dim (int, optional): The last dimension to consider. If None, uses the last dimension of the array.
 |      
 |      Returns:
 |      - HashSort: The current HashSort instance.
 |  
 |  get_all_values(self, start_dim=0, end_dim=-1)
 |      Returns all values based on hash array and specified dimensions.
 |      
 |      Parameters:
 |      - start_dim (int, optional): The starting dimension index to consider.
 |      - end_dim (int, optional): The ending dimension index to consider.
 |      
 |      Returns:
 |      - numpy.ndarray: All values based on the specified dimensions.
 |  
 |  get_shape_array(self, last_dim)
 |      Returns the shape array based on the provided last dimension.
 |      
 |      Parameters:
 |      - last_dim (int): The last dimension to consider.
 |      
 |      Returns:
 |      - Tuple: A tuple containing the shape array and the product of array shape elements from last_dim onwards.
 |  
 |  get_unique_dims_data(self, start_dim=0, end_dim=-1)
 |      Returns unique dimensions data based on hash array.
 |      
 |      Parameters:
 |      - start_dim (int, optional): The starting dimension index to consider.
 |      - end_dim (int, optional): The ending dimension index to consider.
 |      
 |      Returns:
 |      - List[numpy.ndarray]: A list containing unique dimensions data.
 |  
 |  group_equal_values(self, start_dim=0, end_dim=-1)
 |      Groups equal values based on hash array and specified dimensions.
 |      
 |      Parameters:
 |      - start_dim (int, optional): The starting dimension index to consider.
 |      - end_dim (int, optional): The ending dimension index to consider.
 |      
 |      Returns:
 |      - numpy.ndarray: Grouped values based on the specified dimensions.
 |  
 |  sort_by_absolut_index(self, ascending=True)
 |      Sorts the hash array by absolute index (np.flatten()/np.ravel() in ascending or descending order.
 |      
 |      Parameters:
 |      - ascending (bool, optional): If True, sorts in ascending order; otherwise, sorts in descending order.
 |      
 |      Returns:
 |      - numpy.ndarray: The sorted hash array.
 |  
 |  sort_by_hash(self, ascending=False)
 |      Sorts the hash array by hash values in ascending or descending order.
 |      
 |      Parameters:
 |      - ascending (bool, optional): If True, sorts in ascending order; otherwise, sorts in descending order.
 |      
 |      Returns:
 |      - numpy.ndarray: The sorted hash array.
 |  
 |  sort_by_quantity(self, ascending=False)
 |      Sorts the hash array by quantity values in ascending or descending order.
 |      
 |      Parameters:
 |      - ascending (bool, optional): If True, sorts in ascending order; otherwise, sorts in descending order.
 |      
 |      Returns:
 |      - numpy.ndarray: The sorted hash array.
```