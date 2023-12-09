import os
import subprocess
import sys
from cythonanyarray import get_iterarray_shape, get_iterarray, get_pointer_array
import numpy as np


def _dummyimport():
    import Cython


try:
    from .hasharry import xxhash_cython5
except Exception as e:
    cstring = r"""# distutils: language=c++
# distutils: extra_compile_args=/std:c++20 /openmp 
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# distutils: sources = xxhash.c
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: language_level=3
# cython: initializedcheck=False
#include xxhash.h

from cython.parallel cimport prange
cimport cython
import numpy as np
cimport numpy as np
import cython
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
np.import_array()

cdef extern from "xxhash.h":
    ctypedef unsigned long long XXH64_hash_t
    cdef XXH64_hash_t XXH64(void* input, Py_ssize_t length, XXH64_hash_t seed) nogil

cpdef void xxhash_cython5(cython.Py_ssize_t[:,:]  resultdata,Py_ssize_t[:,:]rax,cython.uchar[:] pointerdata,
Py_ssize_t productshape,  cython.Py_ssize_t steps=1,  cython.int seed=0,  cython.int itemsize=1):
    cdef unordered_set[Py_ssize_t] mySet
    cdef unordered_map[Py_ssize_t, Py_ssize_t] my_map
    cdef Py_ssize_t lresultdata =len(resultdata)
    my_map.reserve(lresultdata)
    mySet.reserve(lresultdata)
    cdef unordered_set[Py_ssize_t].iterator it
    cdef Py_ssize_t lenofarray=rax.shape[0]
    cdef Py_ssize_t multipl=pointerdata.shape[0]//productshape
    cdef Py_ssize_t newsteps=steps*multipl
    cdef Py_ssize_t  abs_index
    cdef Py_ssize_t  itemsizex =itemsize*newsteps
    cdef cython.bint isthere
    cdef Py_ssize_t l
    for l in prange(lenofarray,nogil=True):
        abs_index=rax[l][0]
        resultdata[l][0]=abs_index
        resultdata[l][4] = XXH64(&pointerdata[abs_index*multipl], itemsizex, seed)

        isthere= mySet.contains( resultdata[l][4])
        if not isthere:
            resultdata[l][2]=1
            with gil:
                mySet.insert(resultdata[l][4])
                my_map[resultdata[l][4]]=0
        with gil:
            my_map[resultdata[l][4]]+=1
        resultdata[l][3]=my_map[resultdata[l][4]]
        resultdata[l][1]=l


"""
    pyxfile = f"hasharry.pyx"
    pyxfilesetup = f"hasharryarraycompiled_setup.py"

    dirname = os.path.abspath(os.path.dirname(__file__))
    pyxfile_complete_path = os.path.join(dirname, pyxfile)
    pyxfile_setup_complete_path = os.path.join(dirname, pyxfilesetup)

    if os.path.exists(pyxfile_complete_path):
        os.remove(pyxfile_complete_path)
    if os.path.exists(pyxfile_setup_complete_path):
        os.remove(pyxfile_setup_complete_path)
    with open(pyxfile_complete_path, mode="w", encoding="utf-8") as f:
        f.write(cstring)
    numpyincludefolder = np.get_include()
    compilefile = (
            """
	from setuptools import Extension, setup
	from Cython.Build import cythonize
	ext_modules = Extension(**{'py_limited_api': False, 'name': 'hasharry', 'sources': ['hasharry.pyx'], 'include_dirs': [\'"""
            + numpyincludefolder
            + """\'], 'define_macros': [], 'undef_macros': [], 'library_dirs': [], 'libraries': [], 'runtime_library_dirs': [], 'extra_objects': [], 'extra_compile_args': [], 'extra_link_args': [], 'export_symbols': [], 'swig_opts': [], 'depends': [], 'language': None, 'optional': None})

	setup(
		name='hasharry',
		ext_modules=cythonize(ext_modules),
	)
			"""
    )
    with open(pyxfile_setup_complete_path, mode="w", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [x.lstrip().replace(os.sep, "/") for x in compilefile.splitlines()]
            )
        )
    subprocess.run(
        [sys.executable, pyxfile_setup_complete_path, "build_ext", "--inplace"],
        cwd=dirname,
        shell=True,
        env=os.environ.copy(),
    )
    try:
        from .hasharry import xxhash_cython5


    except Exception as fe:
        sys.stderr.write(f'{fe}')
        sys.stderr.flush()


class HashSort:
    r"""
    The HashSort class is designed to perform sorting and grouping operations on multi-dimensional NumPy arrays
    using a hash-based algorithm. It utilizes the xxhash https://xxhash.com/ function (Cython! Not Python!) for efficient hash computation.

    Parameters:
    - a (numpy.ndarray): The input multi-dimensional NumPy array.
    - unordered (bool): If True, will create the index array with multi processing

    Methods:
    - generate_hash_array(last_dim=None): Generates a hash array based on the provided array and optional last_dim.
    - sort_by_absolut_index(ascending=True): Sorts the hash array by absolute index in ascending or descending order.
    - get_unique_dims_data(start_dim=0, end_dim=-1): Returns unique dimensions data based on hash array.
    - get_all_values(start_dim=0, end_dim=-1): Returns all values based on hash array and specified dimensions.
    - group_equal_values(start_dim=0, end_dim=-1): Groups equal values based on hash array and specified dimensions.
    - sort_by_hash(ascending=False): Sorts the hash array by hash values in ascending or descending order.
    - sort_by_quantity(ascending=False): Sorts the hash array by quantity values in ascending or descending order.


    """
    def __init__(self, a, unordered=True):
        r"""
        Initializes a new instance of the HashSort class.

        Parameters:
        - a (numpy.ndarray): The input multi-dimensional NumPy array.
        - unordered (bool): If True, will create the index array with multi processing
        """
        if not a.flags['C_CONTIGUOUS']:
            self.a = np.ascontiguousarray(a)
        else:
            self.a = a
        self.dtype = a.dtype
        self.dty = np.ctypeslib.as_ctypes_type(self.dtype)
        self.b = a.ctypes.data
        self.itsz = a.itemsize
        self.unordered = unordered
        self.iterray = get_iterarray(self.a, dtype=np.int64, unordered=unordered)
        self.pointerdata = get_pointer_array(self.a).view('V1').view(np.uint8)
        self.productshape = np.product(self.a.shape)
        self.rax = None
        self.steps = None
        self.resultdata = None
        self.parsed_data = None
        self.last_dim = None

    def __str__(self):
        return str(self.iterray)

    def __repr__(self):
        return self.__str__()

    def get_shape_array(self, last_dim):
        r"""
        Returns the shape array based on the provided last dimension.

        Parameters:
        - last_dim (int): The last dimension to consider.

        Returns:
        - Tuple: A tuple containing the shape array and the product of array shape elements from last_dim onwards.
        """
        return get_iterarray_shape(self.iterray, last_dim), np.product(self.a.shape[last_dim - 1:])

    def generate_hash_array(self, last_dim=None):
        r"""
        Generates a hash array based on the provided array and optional last_dim.

        Parameters:
        - last_dim (int, optional): The last dimension to consider. If None, uses the last dimension of the array.

        Returns:
        - HashSort: The current HashSort instance.
        """

        if last_dim is None:
            last_dim = len(self.a.shape) - 1
        self.last_dim = last_dim
        self.rax, self.steps = self.get_shape_array(self.last_dim)
        raxlen = len(self.rax)
        self.resultdata = np.zeros((raxlen, 5), dtype=np.int64)
        xxhash_cython5(self.resultdata, self.rax, self.pointerdata, self.productshape, self.steps, 0, 1)
        self.parsed_data = self.resultdata[:raxlen]
        self.parsed_data = self.sort_by_absolut_index()
        return self

    def sort_by_absolut_index(self, ascending=True):
        r"""
        Sorts the hash array by absolute index (np.flatten()/np.ravel() in ascending or descending order.

        Parameters:
        - ascending (bool, optional): If True, sorts in ascending order; otherwise, sorts in descending order.

        Returns:
        - numpy.ndarray: The sorted hash array.
        """
        if ascending:
            return self.parsed_data[np.argsort(self.parsed_data[..., 0], axis=0, kind='stable')]
        return self.parsed_data[np.argsort(self.parsed_data[..., 0], axis=0, kind='stable')][::-1]

    def get_unique_dims_data(self, start_dim=0, end_dim=-1):
        r"""
        Returns unique dimensions data based on hash array.

        Parameters:
        - start_dim (int, optional): The starting dimension index to consider.
        - end_dim (int, optional): The ending dimension index to consider.

        Returns:
        - List[numpy.ndarray]: A list containing unique dimensions data.
        """
        unid = self.parsed_data[np.nonzero(self.parsed_data[..., 2])][..., 0]
        return [self.a[*self.iterray[x][1 + start_dim:end_dim]] for x in unid]

    def get_all_values(self, start_dim=0, end_dim=-1):
        r"""
        Returns all values based on hash array and specified dimensions.

        Parameters:
        - start_dim (int, optional): The starting dimension index to consider.
        - end_dim (int, optional): The ending dimension index to consider.

        Returns:
        - numpy.ndarray: All values based on the specified dimensions.
        """

        return np.vstack([self.a[*y] for y in self.iterray[self.parsed_data[..., 0]][..., start_dim + 1:end_dim]])

    def group_equal_values(self, start_dim=0, end_dim=-1):
        r"""
        Groups equal values based on hash array and specified dimensions.

        Parameters:
        - start_dim (int, optional): The starting dimension index to consider.
        - end_dim (int, optional): The ending dimension index to consider.

        Returns:
        - numpy.ndarray: Grouped values based on the specified dimensions.
        """

        return np.asarray(
            [self.a[*y] for y in self.iterray[np.argsort(self.parsed_data[..., 4], axis=0, kind='stable')][...,
                                 1 + start_dim:end_dim]])


    def sort_by_hash(self, ascending=False):
        r"""
        Sorts the hash array by hash values in ascending or descending order.

        Parameters:
        - ascending (bool, optional): If True, sorts in ascending order; otherwise, sorts in descending order.

        Returns:
        - numpy.ndarray: The sorted hash array.
        """
        if not ascending:
            self.parsed_data[np.argsort(self.parsed_data[..., 4], axis=0, kind='stable')]
        return self.parsed_data[np.argsort(self.parsed_data[..., 4], axis=0, kind='stable')][::-1]

    def sort_by_quantity(self, ascending=False):
        r"""
        Sorts the hash array by quantity values in ascending or descending order.

        Parameters:
        - ascending (bool, optional): If True, sorts in ascending order; otherwise, sorts in descending order.

        Returns:
        - numpy.ndarray: The sorted hash array.
        """
        if not ascending:
            return self.parsed_data[np.argsort(self.parsed_data[..., 3], axis=0, kind='stable')[::-1]]
        return self.parsed_data[np.argsort(self.parsed_data[..., 3], axis=0, kind='stable')]
